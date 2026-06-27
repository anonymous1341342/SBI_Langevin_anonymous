from utils_queuing_single import *
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


def cal_SS(x):
    """
        sample mean and std
    """
    # sample std fits the normal assumption better than sample var
    return torch.cat((x.mean(dim = 0), x.std(dim = 0)), dim=0)
    # return torch.cat((x.mean(dim = 0), x.var(dim = 0)), dim=0)


def BSL_queuing(x_obs, theta_init, prop_scale, n_simu, maxiter):
    """
        Choose sample mean and std as the summary statistic
    """
    # Input:
    # theta_init: initial theta
    # prop_scale: std of the multivariate normal when proposing the next sample in M-H, i.e. theta1 = theta1 + prop_scale \cdot N(0, I)
    # n_simu: the number of simulated datasets for estimating the mean and covariance of the synthetic likelihood
    # maxiter: length of the markov chain
    
    # Output:
    # The trace of the sampling
    x_obs = x_obs.to(device)
    SS_obs = cal_SS(x_obs)

    theta0 = theta_init.clone().to(device)
    theta_path = [theta0]

    # calculate the likelihood at theta0
    # First: obtain the Summary statistics over n_simu simulated samples
    SS_set0 = torch.zeros(n_simu, 10).to(device)
    for j in range(n_simu):
        # generate simulated data
        x = gen_obs_data(theta0[0].item(), theta0[1].item(), theta0[2].item(), dim = 5, obs_size = 500)
        SS_set0[j] = cal_SS(x)
    # Second: calculate the mean and covariance (of the mvn)
    mu0 = SS_set0.mean(dim = 0)
    Cov0 = 1/(n_simu-1) * ( SS_set0 - mu0.view(1, -1).repeat(n_simu, 1) ).T @ ( SS_set0 - mu0.view(1, -1).repeat(n_simu, 1) )

    # iteration starts
    for iter in range(maxiter):
        while True: # truncated normal
            theta_prop = theta0 + prop_scale * torch.randn(theta0.shape).to(device) # propose
            if theta_prop[0] > a1 and theta_prop[0] < b1 and theta_prop[1] > a2 and theta_prop[1] < b2 and theta_prop[2] > a3 and theta_prop[2] < b3:
                break
                
        # calculate the likelihood at theta_prop
        SS_set_prop = torch.zeros(n_simu, 10).to(device)
        for j in range(n_simu):
            # generate simulated data
            x = gen_obs_data(theta_prop[0].item(), theta_prop[1].item(), theta_prop[2].item(), dim = 5, obs_size = 500)
            SS_set_prop[j] = cal_SS(x)
        mu_prop = SS_set_prop.mean(dim = 0)
        Cov_prop = 1/(n_simu-1) * ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) ).T @ ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) )

        acc_prob = torch.linalg.det(Cov0)**(1/2) / torch.linalg.det(Cov_prop)**(1/2) * torch.exp(
            -0.5 * (SS_obs - mu_prop) @ torch.linalg.solve(Cov_prop, SS_obs - mu_prop) + 0.5 * (SS_obs - mu0) @ torch.linalg.solve(Cov0, SS_obs - mu0))
        
        # decide whether to accept the proposal
        if torch.rand(1) <= acc_prob.cpu(): # accept
            # print("accept")
            theta_path.append(theta_prop.clone())
            theta0 = theta_prop.clone()
            mu0 = mu_prop
            Cov0 = Cov_prop
        else: # do not accept
            # print("reject")
            theta_path.append(theta0.clone())
    return theta_path    



def main(obs_id):
    x_obs = pd.read_csv(f"data_obs/x_obs_task{obs_id}.csv")
    x_obs = torch.tensor(x_obs.values, dtype = torch.float32).contiguous()


    theta1 = np.random.uniform(low = a1, high = b1, size = 10000)
    theta2 = np.random.uniform(low = a2, high = b2, size = 10000)
    theta3 = np.random.uniform(low = a3, high = b3, size = 10000)

    theta_r0 = np.c_[theta1, theta2, theta3]
    theta_r0 = torch.tensor(theta_r0, dtype = torch.float32)

    theta_all = []
    num_chains = 0
    while True:
        theta_init = theta_r0[np.random.randint(0, theta_r0.shape[0])]
        if theta_init[2] < 0.05:
            theta_init[2] += 0.05
        prop_scale = 0.1 * theta_r0.std(dim = 0).to(device) # 0.1, 0.05
        n_simu = 100
        maxiter = 3000 
        theta_path = BSL_queuing(x_obs, theta_init, prop_scale, n_simu, maxiter)
        theta_path = torch.stack(theta_path, dim=0)
        num_unique = torch.unique(theta_path, dim=0).size(0)
        if num_unique >= 200:
            theta_all.append(theta_path[1001:].clone().cpu())
            num_chains += 1
            print(f"Have got {num_chains} chains")
        else:
            print(f"This chain failed, with {num_unique} samples")

        if num_chains >= 10:
            break

    theta_all = torch.cat(theta_all, dim=0)

    # create the folder "res" if it does not exist
    Path("res").mkdir(exist_ok=True)

    # save the result
    df = pd.DataFrame( theta_all.cpu().numpy() )
    df.to_csv(f"res/BSL_x_obs_{obs_id}.csv", index=False)




if __name__ == "__main__":
    obs_id = sys.argv[1]
    main(obs_id)