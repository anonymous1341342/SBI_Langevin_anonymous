from utils_nmodel import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma = 0.1 
obs_size = 1000
# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0


def main(task_id):
    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous()

    x_obs = data_obs[:, 0]
    y_obs = data_obs[:, 1]


    psi = get_psi(x_obs, M)
    A = get_A(M)
    # OLS
    design = psi @ torch.linalg.inv(A)
    OLS = torch.linalg.solve(design.T @ design, design.T @ y_obs)

    t1 = time.time()
    ####################
    #  Gibbs sampling  #
    ####################
    def get_cond_mean_and_var(idx, theta, mu, Sigma):
        """
        idx starts from 0
            idx: index that we want to get the conditional distribution of
            mu: mean vector of the whole multivariate normal random vector
            Sigma: Covaraince of the whole multivariate normal random vector
        """
        mask1 = torch.zeros(mu.shape[0])
        mask1[idx] = 1
        mask1 = mask1.bool()

        mask2 = torch.ones(mu.shape[0])
        mask2[idx] = 0
        mask2 = mask2.bool()

        mu1 = mu[idx].item()
        mu2 = mu[mask2]
        Sigma11 = Sigma[idx, idx].item()
        Sigma12 = Sigma[mask1][:, mask2]
        Sigma22 = Sigma[mask2][:, mask2]

        # conditional mean and std
        mu_bar = mu1 + Sigma12 @ torch.linalg.solve(Sigma22, theta[mask2] - mu2).view(-1, 1)
        var_bar = ( Sigma11 - Sigma12 @ torch.linalg.solve(Sigma22, Sigma12.ravel()).view(-1, 1) ).clamp(min = 0.0)
        std_bar = var_bar.sqrt()
        
        return mu_bar.ravel(), std_bar.ravel()

    def Gibbs_BP(theta_init, mu, Sigma, maxiter):
        # dist = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0) # for inverse sampling from a 1-d truncated normal
        res = torch.zeros(maxiter, M + 1)
        theta_curr = theta_init.clone()
        for i in range(maxiter):
            if i % 10000 == 0:
                print(f"iter {i}/{maxiter}")
            # draw one coordinate at one time
            for idx in range(M + 1):           
                mu_bar, sigma_bar = get_cond_mean_and_var(idx, theta_curr, mu, Sigma)
                if idx == 0:
                    prior_u = b0
                    prior_l = a0
                else:
                    prior_u = b
                    prior_l = a
                # # draw from the conditional distribution, which is a 1-dimensional truncated normal

                for k in range(100):
                    prop = mu_bar.item() + sigma_bar.item() * torch.randn(1)
                    if prop > prior_l and prop < prior_u:
                        break
                    if k == 99:
                        # print(f"maxiter 100 reached with mu_bar = {mu_bar}, sigma_bar = {sigma_bar}, index = {idx}")
                        prop = prop.clamp(min = prior_l, max = prior_u)
                theta_curr[idx] = prop
                # print(theta_curr)
            res[i] = theta_curr.clone()
        return res




    n_chain = 10
    maxiter = 40000
    samples_gibbs = torch.zeros(n_chain * 10000, M + 1)
    for i_chain in range(n_chain):
        theta_init = 0.1 + torch.rand(M + 1) * 0.2
        mu = OLS 
        Sigma = sigma**2 * torch.linalg.inv(design.T @ design)
        samples_tmp = Gibbs_BP(theta_init, mu, Sigma, maxiter)
        print(samples_tmp[10000::3].shape)
        samples_gibbs[(i_chain*10000):((i_chain + 1) * 10000)] = samples_tmp[10000::3]

    A = get_A(M)
    grids = torch.arange(0, 1.01, 0.01)
    psi_grids = get_psi(grids, M)

    pred_ys_gibbs = psi_grids @ torch.linalg.inv(A) @ samples_gibbs.T 

    # ensure the folder exists
    os.makedirs('sample_res', exist_ok=True)
    
    np.save(f'sample_res/theta_gibbs_task{task_id}.npy', samples_gibbs.cpu().numpy().astype(np.float32))
    pd.DataFrame( pred_ys_gibbs.numpy() ).to_csv(f'sample_res/pred_ys_gibbs_task{task_id}.csv', index=False, header=False)

    t2 = time.time()
    print(f"Gibbs used {(t2-t1)/60} mins")



if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)