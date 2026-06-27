import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
from pathlib import Path

import pandas as pd
import torch
from utils_SDEMEM_realdata import *

from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters


device = torch.device('cpu')


# ===== Setting for real data ===== #
var_name = ['log_m0', 'log_scale', 'log_offset', 'log_sigma', 'mu_delta', 'mu_gamma', 'mu_k', 'mu_t0', 'log_tau_delta', 'log_tau_gamma', 'log_tau_k', 'log_tau_t0']

# The first 8 parameters have normal priors
prior_mean = torch.tensor([5, 1, 3, -1, -1, -5, 0.5, 0], dtype = torch.float32).unsqueeze(0).to(device)
prior_std = torch.tensor([1, 1, 1, 1, 1, 2, 1, 1], dtype = torch.float32).unsqueeze(0).to(device)

# For the last 4 parameters, the precision follows Gamma priors
prior_alpha = torch.tensor([2, 2, 2, 2], dtype = torch.float32).unsqueeze(0).to(device)
prior_beta = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype = torch.float32).unsqueeze(0).to(device)


T = 30
theta_dim = 12
x_dim = 180 
obs_size = 40

task_id = 0



def main():
    theta_dummy = torch.rand(100, theta_dim).to(device)
    x_dummy = torch.rand(100, x_dim).to(device)



    from torch.distributions import Distribution, Normal, Gamma
    class MixedNormalGammaPrior(Distribution):
        """
        12-dimensional prior for theta.

        theta[0:8] ~ independent Normal(prior_mean, prior_std^2)

        For theta[8:12]:
            eta_j = exp(-2 * theta_j) ~ Gamma(alpha_j, beta_j)

        beta is the Gamma rate parameter, consistent with torch.distributions.Gamma.
        """

        arg_constraints = {}
        support = torch.distributions.constraints.real
        has_rsample = False

        def __init__(self, prior_mean, prior_std, prior_alpha, prior_beta, validate_args=False):
            super().__init__(validate_args=validate_args)

            # Robustly convert shapes:
            # (1, 8) or (8,) -> (8,)
            # (1, 4) or (4,) -> (4,)
            self.prior_mean = prior_mean.reshape(-1)
            self.prior_std = prior_std.reshape(-1)
            self.prior_alpha = prior_alpha.reshape(-1)
            self.prior_beta = prior_beta.reshape(-1)

            assert self.prior_mean.shape == (8,), f"prior_mean shape is {self.prior_mean.shape}"
            assert self.prior_std.shape == (8,), f"prior_std shape is {self.prior_std.shape}"
            assert self.prior_alpha.shape == (4,), f"prior_alpha shape is {self.prior_alpha.shape}"
            assert self.prior_beta.shape == (4,), f"prior_beta shape is {self.prior_beta.shape}"

            self.device = self.prior_mean.device
            self.dtype = self.prior_mean.dtype

            self.normal_dist = Normal(self.prior_mean, self.prior_std)
            self.gamma_dist = Gamma(self.prior_alpha, self.prior_beta)

        def sample(self, sample_shape=torch.Size()):
            sample_shape = torch.Size(sample_shape)

            # theta[0:8]
            theta_normal = self.normal_dist.sample(sample_shape)

            # eta = exp(-2 theta) ~ Gamma
            eta = self.gamma_dist.sample(sample_shape)
            theta_gamma = -0.5 * torch.log(eta)

            return torch.cat([theta_normal, theta_gamma], dim=-1)

        def log_prob(self, theta):
            theta = theta.to(device=self.device, dtype=self.dtype)

            theta_normal = theta[..., :8]
            theta_gamma = theta[..., 8:]

            logp_normal = self.normal_dist.log_prob(theta_normal).sum(dim=-1)

            eta = torch.exp(-2.0 * theta_gamma)

            logp_gamma = self.gamma_dist.log_prob(eta).sum(dim=-1)

            log_abs_jacobian = (
                torch.log(torch.tensor(2.0, device=self.device, dtype=self.dtype))
                - 2.0 * theta_gamma
            ).sum(dim=-1)

            return logp_normal + logp_gamma + log_abs_jacobian
        

    prior = MixedNormalGammaPrior(
        prior_mean=prior_mean.unsqueeze(0).to(device),
        prior_std=prior_std.unsqueeze(0).to(device),
        prior_alpha=prior_alpha.unsqueeze(0).to(device),
        prior_beta=prior_beta.unsqueeze(0).to(device),
    )



    from torch.distributions import MultivariateNormal

    # ====== Load the trained NLE model ====== #
    # prior = MultivariateNormal(loc=prior_mean.to(device), covariance_matrix=torch.diag(prior_std**2).to(device))
    # prior.to(device) #inplace
    inference = NLE(prior, show_progress_bars=True, density_estimator="maf", device = device)


    # Build the neural net once so the modules exist before loading state_dict.
    # We use a tiny dummy batch only for initializing the neural networks
    inference.append_simulations(
        theta_dummy.to(device), 
        x_dummy.to(device)).train(
            training_batch_size=10,
            max_num_epochs=1
            )


    weights_path = Path("NLE_model") / f"nle_net_weights_task{task_id}.pth"
    state_dict = torch.load(weights_path, map_location=device)
    inference._neural_net.load_state_dict(state_dict)
    inference._neural_net.eval()


    mcmc_parameters = MCMCPosteriorParameters(
        method='nuts_pyro', 
        num_chains=1,
        thin=1,
        warmup_steps=500,
        init_strategy="proposal"
    )

    posterior = inference.build_posterior(
        posterior_parameters=mcmc_parameters
    )


    # ========== load observed data
    df = pd.read_excel("realdata/20160427_mean_eGFP.xlsx", header=None)
    x_obs = torch.tensor(df.to_numpy(), dtype=torch.float32)[:, 1:].T.log() 
    x_obs = x_obs[:obs_size]

    theta_post = posterior.sample(sample_shape=(1000,), x=x_obs.to(device))

    os.makedirs("res_NLE", exist_ok=True)
    np.save(f"res_NLE/theta_post_sameprior.npy", theta_post.cpu().numpy())


if __name__ == "__main__":
    main()