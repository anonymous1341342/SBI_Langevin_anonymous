from utils_monoBP_single import *
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
import ot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma = 0.1 # noise level
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

    ### generate proposal theta
    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()

    lower = torch.zeros(M + 1) # .to(device)
    lower[0] = a0
    lower[1:] = a
    upper = torch.zeros(M + 1) # .to(device)
    upper[0] = b0
    upper[1:] = b

    actual_inf_rate = torch.ones(M + 1)
    actual_inf_rate[-2:] = 2

    inf_rate = torch.zeros(M + 1)
    for i in range(M + 1):
        inf_rate[i] = get_inf_rate(mode = theta_pre.mean(dim = 0)[i].item(), std_orig = theta_pre.std(dim = 0)[i].item(),
                        lower = lower[i].item(), upper = upper[i].item(), actual_inf_rate = actual_inf_rate[i].item())


    mean_theta = theta_pre.mean(dim = 0)
    std_theta = inf_rate * theta_pre.std(dim = 0)


    sample_size = int(1e5) 
    theta_r0, data_r0 = gen_ref_distinct_theta(mean_theta, std_theta, lower, upper, sample_size + 100)

    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask]
    data_r0 = data_r0[~bad_mask]


    t1 = time.time()
    ###########
    #   BSL   #
    ###########
    def cal_SS(x, y):
        """
            Calculate the sufficient statistics
            consider fixed design
        """
        psi = get_psi(x, M).to(device)
        A = get_A(M).to(device)
        design = psi @ torch.linalg.inv(A)
        return design.T @ y

    def BSL_MH(data_obs, theta_init, prop_scale, n_simu, maxiter):
        """
            Metropolis-Hasting sampling, with the synthetic likelihood
        """
        # Input:
        # theta_init: initial theta
        # prop_scale: std of the multivariate normal when proposing the next sample in M-H, i.e. theta1 = theta1 + prop_scale \cdot N(0, I)
        # n_simu: the number of simulated datasets for estimating the mean and covariance of the synthetic likelihood
        # maxiter: length of the markov chain
        
        # Output:
        # The trace of the sampling
        data_obs = data_obs.to(device)
        SS_obs = cal_SS(data_obs[:, 0], data_obs[:, 1])

        theta0 = theta_init.clone().to(device)
        theta_path = [theta0]

        # calculate the likelihood at theta0
        # First: obtain the Summary statistics over n_simu simulated samples
        SS_set0 = torch.zeros(n_simu, theta_init.shape[0]).to(device)
        for j in range(n_simu):
            # generate simulated data
            x = torch.rand(obs_size).to(device)
            A = get_A(M).to(device)
            psi = get_psi(x, M).to(device)
            y = psi @ torch.linalg.inv(A) @ theta0 + sigma * torch.randn(obs_size).to(device)
            SS_set0[j] = cal_SS(x, y)
        # Second: calculate the mean and covariance (of the mvn)
        mu0 = SS_set0.mean(dim = 0)
        Cov0 = 1/(n_simu-1) * ( SS_set0 - mu0.view(1, -1).repeat(n_simu, 1) ).T @ ( SS_set0 - mu0.view(1, -1).repeat(n_simu, 1) )

        # iteration starts
        for iter in range(maxiter):
            theta_prop = theta0 + prop_scale * torch.randn(theta0.shape).to(device) # propose
            # calculate the likelihood at theta_prop
            SS_set_prop = torch.zeros(n_simu, theta_init.shape[0]).to(device)
            for j in range(n_simu):
                # generate simulated data
                x = torch.rand(obs_size).to(device)
                A = get_A(M).to(device)
                psi = get_psi(x, M).to(device)
                y = psi @ torch.linalg.inv(A) @ theta_prop + sigma * torch.randn(obs_size).to(device)
                SS_set_prop[j] = cal_SS(x, y)
            mu_prop = SS_set_prop.mean(dim = 0)
            Cov_prop = 1/(n_simu-1) * ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) ).T @ ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) )

            acc_prob = torch.linalg.det(Cov0)**(1/2) / torch.linalg.det(Cov_prop)**(1/2) * torch.exp(
                -0.5 * (SS_obs - mu_prop) @ torch.linalg.solve(Cov_prop, SS_obs - mu_prop) + 0.5 * (SS_obs - mu0) @ torch.linalg.solve(Cov0, SS_obs - mu0))

            if torch.rand(1) <= acc_prob.cpu(): # accept
                # print("accept")
                theta_path.append(theta_prop.clone())
                theta0 = theta_prop.clone()
                mu0 = mu_prop
                Cov0 = Cov_prop
            else: 
                # print("reject")
                theta_path.append(theta0.clone())
        return theta_path    



    theta_all = []
    num_chains = 0
    while True:
        theta_init = theta_r0[np.random.randint(0, theta_r0.shape[0])]
        prop_scale = 0.1 * theta_r0.std(dim = 0).to(device)
        n_simu = 100
        maxiter = 1200
        
        theta_path = BSL_MH(data_obs, theta_init, prop_scale, n_simu, maxiter)
        theta_path = torch.stack(theta_path, dim=0)
        num_unique = torch.unique(theta_path, dim=0).size(0)
        if num_unique >= 100:
            theta_all.append(theta_path[200:].clone()) # 200 burn-in
            num_chains += 1
            print(f"Have got {num_chains} chains")
        else:
            print(f"This chain failed, with {num_unique} samples")

        if num_chains >= 10:
            break
            
    theta_all = torch.cat(theta_all, dim=0)

    A = get_A(M)
    grids = torch.arange(0, 1.01, 0.01)
    psi_grids = get_psi(grids, M)
    pred_ys_BSL = psi_grids @ torch.linalg.inv(A) @ theta_all.T.cpu()


    # ensure the directory exists
    os.makedirs('sample_res_others', exist_ok=True)

    pd.DataFrame( pred_ys_BSL.numpy() ).to_csv(f'sample_res_others/pred_ys_BSL_task{task_id}.csv', index=False, header=False)
    t2 = time.time()
    print(f"BSL used {(t2-t1)/60} mins")



if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
