from utils_sm import *
from utils_SDEMEM import *
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


# ===== Setting ===== #
obs_size = 200
T = 30
theta_dim = 12
x_dim = 180

prior_mean = torch.tensor([5, 1, 3, -1.5, -0.694, -3, 0.027, 0, -0.8, -0.8, -0.8, -0.8], dtype = torch.float32)
prior_std = torch.tensor([1, 1, 1, 1, 0.6, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype = torch.float32)

# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
theta_true = torch.tensor([5.7, 0.7, 2.08, -1.6, -0.694, -3, 0.027, 0, -1.15, -1.15, -1.15, -1.15], dtype = torch.float32).reshape(1, -1)


var_name = ['log_m0', 'log_scale', 'log_offset', 'log_sigma', 'mu_delta', 'mu_gamma', 'mu_k', 'mu_t0', 'log_tau_delta', 'log_tau_gamma', 'log_tau_k', 'log_tau_t0']


def main(obs_id):
    sm_rd = 1

    task_id = obs_id % 10 # 1 model takes 10 obsesrved data

    # ======= Load SW data ======= #
    theta_SW1 = np.load(f"res_SW1/theta_SW1_task{task_id}.npy")[:100]
    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)

    prop_mean = theta_SW1.mean(dim = 0, keepdims = True)
    prop_std = theta_SW1.std(dim = 0, keepdims = True) 

    # inflate the proposal std
    prop_std *= 2

    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")



    # ======== Load the SingleModel ======== #
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

    print(scale_score)

    # ======== Load the debreg model ======= #
    save_path = Path(f"DebRegModel_fisher/sm_round{1}/checkpoint_task{task_id}.pth")
    DebReg_model = Deb_ELU_sparse(input_dim = theta_dim, output_dim = theta_dim, hidden_size = 128, num_layers = 3).to(device)

    checkpoint = torch.load(save_path, map_location = device)
    DebReg_model.load_state_dict(checkpoint['model_state_dict'])


    def NScore_DebReged(theta, x): # the input x is "N-data"
        return model.cal_penalty(theta, x).sum(dim = 1) - DebReg_model(theta) * obs_size




    # ======= Sampling: draw multiple chains ======== #
    def proj(theta):
        theta = torch.clamp(theta, min=-3, max=3)
        return theta

    def proj_draw_post_vec(x_obs, theta_init, epis, S):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_chain = theta_init.shape[0]
        
        theta0 = theta_init.to(device) # initial value of theta
        x_obs = x_obs.to(device).view(1, -1)
        epis = epis.to(device).view(1, -1)
        samples = torch.zeros(n_chain, S, theta_dim)

        prior_mean_norm = ( (prior_mean - prop_mean) / prop_std ).to(device)
        prior_std_norm = ( prior_std / prop_std ).to(device)
        for i in range(S):
            like_score_hat = NScore_DebReged(theta0, x_obs.repeat(n_chain, 1)).detach().to(device) # [n_chain, theta_dim]
            # prior_score = 0.0
            prior_score = (prior_mean_norm - theta0) / (prior_std_norm**2) # [n_chain, theta_dim]
            theta1 = theta0 + epis * (like_score_hat + prior_score) + torch.sqrt(2.0 * epis) * torch.randn(theta0.shape).to(device) # draw a new sample
            theta1 = proj(theta1)
            theta0 = theta1 # use theta1 as the initial value of the next iteration
            
            samples[:, i, :] = theta1.cpu().clone()
        return samples


    # Load x_obs
    x_obs = torch.from_numpy(np.load(f"data_obs/x_obs_task{obs_id}.npy"))

    x_obs = (x_obs - mean_x) / std_x
    x_obs = x_obs[:obs_size, :]
    print(f"x_obs.shape = {x_obs.shape}")


    # Sampling starts
    n_chain = 100
    theta_init = torch.zeros(n_chain, theta_dim)  

    epis = 1e-2 / obs_size * torch.ones_like(scale_score) 
    for idx in [7, 11]:
        epis[idx] *= 10

    S = 20000 
    samples_norm = proj_draw_post_vec(x_obs, theta_init, epis, S)
    samples = samples_norm * prop_std + prop_mean

    print(f"samples.shape = {samples.shape}")


    # Save the samples
    save_dir = Path(f"res_inference/sm_round{sm_rd}")
    save_dir.mkdir(parents=True, exist_ok=True)  
    np.save(save_dir / f"samples{obs_id}.npy", samples.detach().cpu().numpy())



if __name__ == "__main__":
    obs_id = int(sys.argv[1])
    main(obs_id)