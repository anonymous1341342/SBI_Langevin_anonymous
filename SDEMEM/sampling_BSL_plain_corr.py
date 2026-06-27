from utils_SDEMEM import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path
import torch
import ot


task_id = 0 # int(sys.argv[1])

maxiter = 10000 # int(config["maxiter"]) 
chain_id = 0 # int(config["chain_id"]) # 0, 1, 2, 3, 4



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Setting ===== #
obs_size = 200
T = 30
theta_dim = 12
x_dim = 180

prior_mean = torch.tensor([5, 1, 3, -1.5, -0.694, -3, 0.027, 0, -0.8, -0.8, -0.8, -0.8], dtype = torch.float32)
prior_std = torch.tensor([1, 1, 1, 1, 0.6, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype = torch.float32)

# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
theta_true = torch.tensor([5.7, 0.7, 2.08, -1.6, -0.694, -3, 0.027, 0, -1.15, -1.15, -1.15, -1.15], dtype = torch.float32).reshape(1, -1)


def main():
    def compute_acf_lag(x_centered: torch.Tensor, lag: int, eps: float = 1e-8) -> torch.Tensor:
        T = x_centered.shape[-1]
        if lag <= 0 or lag >= T:
            raise ValueError(f"lag must be between 1 and T-1, got lag={lag}, T={T}")

        numer = (x_centered[..., :-lag] * x_centered[..., lag:]).sum(dim=-1)
        denom = (x_centered ** 2).sum(dim=-1) + eps
        return numer / denom


    def series_level_summary_batched(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (MC_size, n_series, T), got {tuple(x.shape)}")

        T = x.shape[-1]
        if T <= 12:
            raise ValueError(f"Need T > 12 because ACF(12) is used, got T={T}")

        mean_x = x.mean(dim=-1)
        x_centered = x - mean_x.unsqueeze(-1)

        var_x = (x_centered ** 2).mean(dim=-1)
        sd_x = torch.sqrt(var_x + eps)

        z = x_centered / (sd_x.unsqueeze(-1) + eps)
        skew_x = (z ** 3).mean(dim=-1)
        kurt_x = (z ** 4).mean(dim=-1)

        dx = x[..., 1:] - x[..., :-1]
        mean_dx2 = (dx ** 2).mean(dim=-1)

        acf_1 = compute_acf_lag(x_centered, lag=1, eps=eps)
        acf_2 = compute_acf_lag(x_centered, lag=2, eps=eps)
        acf_3 = compute_acf_lag(x_centered, lag=3, eps=eps)
        acf_6 = compute_acf_lag(x_centered, lag=6, eps=eps)
        acf_12 = compute_acf_lag(x_centered, lag=12, eps=eps)

        summary = torch.stack(
            [mean_x, sd_x, skew_x, kurt_x, mean_dx2, acf_1, acf_2, acf_3, acf_6, acf_12],
            dim=-1,
        )
        return summary


    def dataset_level_summary_from_series_batch(x: torch.Tensor, eps: float = 1e-8, unbiased_std: bool = False) -> torch.Tensor:
        series_summary_batch = series_level_summary_batched(x, eps=eps)
        mean_summary = series_summary_batch.mean(dim=1)
        std_summary = series_summary_batch.std(dim=1, unbiased=unbiased_std)
        dataset_summary_batch = torch.cat([mean_summary, std_summary], dim=-1)
        return dataset_summary_batch


    def log_prior_diag_gaussian(theta, prior_mean, prior_std):
        theta = theta.to(device)
        prior_mean = prior_mean.to(device)
        prior_std = prior_std.to(device)

        z = (theta - prior_mean) / prior_std
        return (-0.5 * z.pow(2) - torch.log(prior_std) - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=device))).sum()


    def BSL_MH_with_prior_plain_corr(data_obs, theta_init, prop_scale, n_simu, maxiter, prior_mean, prior_std, corr_block, corr_matrix):
        """
        Plain random walk Metropolis-Hastings with Gaussian synthetic likelihood
        and an independent Gaussian prior.

        One 12-dimensional proposal is made per iteration.
        Dims corr_block use a correlated Gaussian proposal; all other dims remain independent.
        """
        start_time = time.time()

        MC_size = n_simu

        data_obs = data_obs.to(device)
        SS_obs = dataset_level_summary_from_series_batch(data_obs.unsqueeze(0)).squeeze(0)

        theta0 = theta_init.clone().to(device)
        theta_path = [theta0.clone()]

        data_simu = gen_x_given_theta(theta0.repeat(obs_size * MC_size, 1), T=T, mute=True).reshape(MC_size, obs_size, -1)
        SS_set0 = dataset_level_summary_from_series_batch(data_simu)
        mu0 = SS_set0.mean(dim=0)
        Cov0 = 1 / (n_simu - 1) * (SS_set0 - mu0.view(1, -1).repeat(n_simu, 1)).T @ (SS_set0 - mu0.view(1, -1).repeat(n_simu, 1))
        logprior0 = log_prior_diag_gaussian(theta0, prior_mean, prior_std)

        ridge = 0.0
        I = torch.eye(Cov0.shape[0], device=device)

        corr_block = list(corr_block)
        corr_matrix = corr_matrix.to(device)
        corr_block_idx = torch.tensor(corr_block, dtype=torch.long, device=device)
        corr_block_scale = prop_scale[corr_block_idx].to(device)
        corr_block_cov = torch.diag(corr_block_scale) @ corr_matrix @ torch.diag(corr_block_scale)
        corr_block_chol = torch.linalg.cholesky(corr_block_cov)

        for iter in range(maxiter):
            if (iter + 1) % 100 == 0:
                print(f"Iter {iter + 1}/{maxiter}, time elapsed: {time.time() - start_time:.2f} seconds")

            theta_prop = theta0 + prop_scale * torch.randn(theta0.shape, device=device)
            theta_prop[corr_block_idx] = theta0[corr_block_idx] + corr_block_chol @ torch.randn(corr_block_idx.numel(), device=device)

            data_simu_prop = gen_x_given_theta(theta_prop.repeat(obs_size * MC_size, 1), T=T, mute=True).reshape(MC_size, obs_size, -1)
            SS_set_prop = dataset_level_summary_from_series_batch(data_simu_prop)

            mu_prop = SS_set_prop.mean(dim=0)
            Cov_prop = 1 / (n_simu - 1) * (SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1)).T @ (SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1))
            logprior_prop = log_prior_diag_gaussian(theta_prop, prior_mean, prior_std)

            Cov0_reg = Cov0 + ridge * I
            Cov_prop_reg = Cov_prop + ridge * I

            sign0, logdet0 = torch.linalg.slogdet(Cov0_reg)
            signp, logdetp = torch.linalg.slogdet(Cov_prop_reg)

            if sign0 <= 0 or signp <= 0:
                log_acc_prob = torch.tensor(float('-inf'), device=device)
            else:
                diff0 = SS_obs - mu0
                diffp = SS_obs - mu_prop

                quad0 = diff0 @ torch.linalg.solve(Cov0_reg, diff0)
                quadp = diffp @ torch.linalg.solve(Cov_prop_reg, diffp)

                loglik0 = -0.5 * logdet0 - 0.5 * quad0
                loglikp = -0.5 * logdetp - 0.5 * quadp

                logpost0 = loglik0 + logprior0
                logpostp = loglikp + logprior_prop
                log_acc_prob = logpostp - logpost0

            if torch.log(torch.rand(1, device=device)) <= log_acc_prob:
                theta0 = theta_prop.clone()
                mu0 = mu_prop
                Cov0 = Cov_prop
                logprior0 = logprior_prop

            theta_path.append(theta0.clone())

        return theta_path


    # Load SW data
    theta_SW1 = np.load(f"res_SW1/theta_SW1_task{task_id}.npy")[:100]
    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)

    prop_mean = theta_SW1.mean(dim = 0, keepdims = True)



    # ====== Sampling begins ====== #
    corr_block_016 = [0, 1, 6]
    corr_matrix_016 = torch.tensor([
        [1.0, 0.0, -0.8],
        [0.0, 1.0, -0.5],
        [-0.8, -0.5, 1.0],
    ], dtype=torch.float32)

    # theta_init = theta_true.clone().ravel().to(device)

    theta_init = prop_mean.clone().ravel().to(device)
    prop_scale = 0.1 * torch.tensor([
        0.5481, 0.3472, 0.0058, 0.0052, 0.0264, 0.0225, 0.6634, 0.0227, 0.0604,
        0.0559, 0.0568, 0.0517,
    ], dtype=torch.float32).to(device)

    n_simu = 200
    maxiter = 10000



    # ========== load observed data
    x_obs = torch.from_numpy(np.load(f"data_obs/x_obs_task{task_id}.npy"))

    print(x_obs.shape)

    theta_path_prior_plain_corr = BSL_MH_with_prior_plain_corr(
        x_obs,
        theta_init,
        prop_scale,
        n_simu,
        maxiter,
        prior_mean,
        prior_std,
        corr_block_016,
        corr_matrix_016,
    )

    theta_path_prior_plain_corr = torch.stack(theta_path_prior_plain_corr, dim=0)


    # ensure the folder exists
    os.makedirs('res_BSL_plain_corr', exist_ok=True)

    np.save(
        f'res_BSL_plain_corr/theta_BSL_task{task_id}_chain{chain_id}.npy',
        theta_path_prior_plain_corr.cpu().numpy(),
    )


if __name__ == "__main__":
    main()