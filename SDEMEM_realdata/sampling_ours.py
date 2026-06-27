from utils_sm import *
from utils_SDEMEM_realdata import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import math
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import time


obs_size = 40

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

task_id = 0
sm_rd = 1

def main():
    # Load SW data
    theta_SW1 = np.load("res_SW1/theta_SW1.npy")
    loss_SW1 = np.load("res_SW1/final_loss.npy")

    nan_idx = np.isnan(theta_SW1).any(axis=1)
    theta_SW1 = theta_SW1[~nan_idx]
    loss_SW1 = loss_SW1[~nan_idx]

    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)[:100]
    print(f"theta_SW1.shape = {theta_SW1.shape}")


    prop_mean = theta_SW1.mean(dim = 0, keepdims = True).to(device)
    prop_std = theta_SW1.std(dim = 0, keepdims = True).to(device)

    # inflate the proposal std
    prop_std *= 2

    # the previous prop_std is too small for these two dimensions
    prop_std[0, 0] *= 3
    prop_std[0, 6] *= 3


    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")


    # Load the SingleModel_init
    checkpoint_path = f"scaled_fishermodel_weighted/sm_round{sm_rd}/checkpoint_task{task_id}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ELU_single_LikeScoreMatchingNN_sparse(theta_dim, x_dim, 128, 3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    bias_lastlayer = checkpoint['bias_lastlayer']

    with torch.no_grad(): 
        model.net[-1].bias -= bias_lastlayer.to(device)



    mean_x = checkpoint['mean_x'].cpu()
    std_x = checkpoint['std_x'].cpu()

    path_val_loss_all_dim = checkpoint['path_val_loss_all_dim']
    path_val_loss_all_dim = np.stack(path_val_loss_all_dim, axis = 0) # shape (num_epochs, theta_dim)
    scale_score = torch.tensor( -path_val_loss_all_dim.min(axis=0), dtype = torch.float32)


    # Load the debreg model
    save_path = Path(f"DebRegModel_fisher/sm_round{1}/checkpoint_task{task_id}.pth")
    DebReg_model = Deb_ELU_sparse(input_dim = theta_dim, output_dim = theta_dim, hidden_size = 128, num_layers = 3).to(device)

    checkpoint = torch.load(save_path, map_location = device)
    DebReg_model.load_state_dict(checkpoint['model_state_dict'])


    def NScore_DebReged(theta, x): # the input x is "N-data"
        return model.cal_penalty(theta, x).sum(dim = 1) - DebReg_model(theta) * obs_size


    # ========== load observed data
    df = pd.read_excel("realdata/20160427_mean_eGFP.xlsx", header=None)
    x_obs_orig = torch.tensor(df.to_numpy(), dtype=torch.float32)[:, 1:].T.log() # original scale, make a copy
    x_obs = torch.tensor(df.to_numpy(), dtype=torch.float32)[:, 1:].T.log()
    x_obs = (x_obs - mean_x) / std_x

    x_obs = x_obs[:obs_size]


    def proj_draw_post_precond_vec(x_obs, theta_init, epis, S):
        """
        Vectorized preconditioned Langevin sampler with multiple chains.

        theta_init: shape (num_chains, theta_dim)
        x_obs: shape (obs_dim,) or (num_chains, obs_dim)
        epis: shape (theta_dim, theta_dim), SPD preconditioning matrix
        returns: tensor of shape (S, num_chains, theta_dim)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        theta0 = theta_init.to(device)
        x_obs = x_obs.to(device).view(1, -1)


        epis = epis.to(device)
        L = torch.linalg.cholesky(epis)

        # to calculate the prior score for the first 8 parameters
        prior_mean_norm = ( (prior_mean - prop_mean[:, :8]) / prop_std[:, :8] ).to(device)
        prior_std_norm = ( prior_std / prop_std[:, :8] ).to(device)

        samples = []

        for _ in tqdm(range(S)):
            like_score_hat = NScore_DebReged(theta0, x_obs.repeat(theta0.shape[0], 1)).detach().to(device)   # (num_chains, theta_dim)
            
            # prior_score = 0.0
            prior_score = torch.zeros_like(like_score_hat).to(device)
            # calculate the prior score for the first 8 parameters
            prior_score[:, :8] = (prior_mean_norm - theta0[:, :8]) / (prior_std_norm**2)

            # calculate the prior score for the last 4 parameters, which have Gamma priors on the precision
            prior_score[:, 8:] = -2 * prior_alpha * prop_std[:, 8:] + 2 * prior_beta * prop_std[:, 8:] * torch.exp(-2 * (prop_mean[:, 8:] + prop_std[:, 8:] * theta0[:, 8:]))

            score = like_score_hat + prior_score
            noise = math.sqrt(2.0) * torch.randn_like(theta0) @ L.T

            theta1 = theta0 + score @ epis + noise
            theta1 = proj(theta1)

            theta0 = theta1
            samples.append(theta1.cpu().clone())

        return torch.stack(samples, dim=0)


    def proj(theta):
        return torch.clamp(theta, min=-4, max=4) 



    num_chains = 10 
    theta_init = torch.zeros(num_chains, theta_dim)

    epis = 1e-2 / obs_size * torch.ones_like(scale_score)

    for idx in [2, 3, 8, 9]:
        epis[idx] *= 0.1

    for idx in [0, 1, 6]:
        epis[idx] *= 2 

    for idx in [7]:
        epis[idx] *= 10


    epis = epis.diag() # make a preconditioning matrix

    epis[0, 6] = torch.sqrt(epis[0, 0] * epis[6, 6]) * (-0.6)
    epis[6, 0] = epis[0, 6]

    epis[1, 6] = torch.sqrt(epis[1, 1] * epis[6, 6]) * (-0.6) 
    epis[6, 1] = epis[1, 6]


    epis[4, 5] = torch.sqrt(epis[4, 4] * epis[5, 5]) * (-0.6) 
    epis[5, 4] = epis[4, 5]

    samples_norm = proj_draw_post_precond_vec(x_obs, theta_init, epis, S=10000)

    samples = samples_norm * prop_std.cpu() + prop_mean.cpu()

    theta_post = samples[2000::10, :, :].reshape(-1, theta_dim)
    np.save(f"sample_res_all/theta_post_precond_mchain_sameprior_task{task_id}.npy", theta_post.cpu().numpy())



if __name__ == "__main__":
    main()