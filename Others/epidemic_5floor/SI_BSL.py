from utils_SI_5F import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import time



# Settings for the floor and room assignments
K = 5 # number of floors
N = 300
NR = 2 # number of people in each room
NF = int(N/K) # number of people on each floor

F_assign = torch.zeros(N, K)
for k in range(K):
    F_assign[(k*NF):((k+1)*NF), k] = 1
C_F = F_assign @ F_assign.T 

R_assign = torch.zeros(N, int(N/NR))
for r in range( int(N/NR) ):
    R_assign[(r*NR):((r+1)*NR), r] = 1
C_R = R_assign @ R_assign.T

F_assign = F_assign.to(device)
C_F = C_F.to(device)
C_R = C_R.to(device)

gamma = 0.05
alpha = 0.1
eta = 0.1 
T = 52

beta_true = torch.tensor([0.05, 0.02, 0.04, 0.06, 0.08, 0.1, 0.05]).to(device)


def main(task_id):
    data_obs = torch.tensor(np.load(f"data_obs/data_obs_task{task_id}.npy"), dtype=torch.float32).to(device)


    def cal_SS(x):
        """
            sample mean and std
        """
        # sample std fits the normal assumption better than sample var
        return torch.cat((x.mean(dim = 0), x.std(dim = 0)), dim=0)
        # return torch.cat((x.mean(dim = 0), x.var(dim = 0)), dim=0)

    def BSL_SI_prior(data_obs, log_theta_init, prop_scale, n_simu, maxiter):
        """
            Choose sample mean and std as the summary statistic
        """
        # Input:
        # log_theta_init: initial log_theta
        # prop_scale: std of the multivariate normal when proposing the next sample in M-H, i.e. theta1 = theta1 + prop_scale \cdot N(0, I)
        # n_simu: the number of simulated datasets for estimating the mean and covariance of the synthetic likelihood
        # maxiter: length of the markov chain
        
        # Output:
        # The trace of the sampling
        data_obs = data_obs.to(device)
        data_obs = get_SS(data_obs).reshape(-1, 52).T
        data_obs = data_obs[:, 1:] # drop the first column to avoid linear dependence
        SS_obs = cal_SS( data_obs ) # sample mean and std

        log_theta0 = log_theta_init.clone().to(device)
        log_theta_path = [log_theta0]

        # calculate the likelihood at log_theta0
        # First: obtain the Summary statistics over n_simu simulated samples
        SS_set0 = torch.zeros(n_simu, 12).to(device)
        for j in range(n_simu):
            # generate simulated data
            z = gen_z(N, T)
            data = m_vec_partial(N, T, log_theta0.exp(), gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
            data = get_SS(data).reshape(-1, 52).T
            data = data[:, 1:] # drop the first column to avoid exact linear dependence
            SS_set0[j] = cal_SS(data)
        # Second: calculate the mean and covariance (of the mvn)
        mu0 = SS_set0.mean(dim = 0)
        Cov0 = 1/(n_simu-1) * ( SS_set0 - mu0.view(1, -1).repeat(n_simu, 1) ).T @ ( SS_set0 - mu0.view(1, -1).repeat(n_simu, 1) )

        # iteration starts
        for iter in range(maxiter):
            log_theta_prop = log_theta0 + prop_scale * torch.randn(log_theta0.shape).to(device) # propose
            # print(f"log_theta_prop = {log_theta_prop}")
                    
            # calculate the likelihood at theta_prop
            SS_set_prop = torch.zeros(n_simu, 12).to(device)
            for j in range(n_simu):
                # generate simulated data
                z = gen_z(N, T)
                data = m_vec_partial(N, T, log_theta_prop.exp(), gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
                data = get_SS(data).reshape(-1, 52).T
                data = data[:, 1:] # drop the first column to avoid exact linear dependence
                SS_set_prop[j] = cal_SS(data)
            mu_prop = SS_set_prop.mean(dim = 0)
            Cov_prop = 1/(n_simu-1) * ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) ).T @ ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) )

            acc_prob = torch.linalg.det(Cov0)**(1/2) / torch.linalg.det(Cov_prop)**(1/2) * torch.exp(
                -0.5 * (SS_obs - mu_prop) @ torch.linalg.solve(Cov_prop, SS_obs - mu_prop) + 0.5 * (SS_obs - mu0) @ torch.linalg.solve(Cov0, SS_obs - mu0))
            acc_prob = acc_prob * (-0.5 * torch.linalg.norm(log_theta_prop + 3.)**2 + 0.5 * torch.linalg.norm(log_theta0 + 3.)**2).exp() # the prior is not uniform

            if torch.rand(1) <= acc_prob.cpu(): # accept
                # print("accept")
                log_theta_path.append(log_theta_prop.clone())
                log_theta0 = log_theta_prop.clone()
                mu0 = mu_prop
                Cov0 = Cov_prop
            else: # do not accept
                # print("reject")
                log_theta_path.append(log_theta0.clone())
        return log_theta_path    


    theta_all = []
    num_chains = 0
    while True:
        start_time = time.time()
        log_theta_init = -3.0 + torch.randn(7)
        prop_scale = 0.1 
        n_simu = 100
        maxiter = 1000 
        
        theta_path = BSL_SI_prior(data_obs, log_theta_init, prop_scale, n_simu, maxiter)
        theta_path = torch.stack(theta_path, dim=0)
        num_unique = torch.unique(theta_path, dim=0).size(0)
        if num_unique >= 80:
            theta_all.append(theta_path[201:].clone()) # 200 burn-in
            num_chains += 1
            print(f"Have got {num_chains} chains. This chain has {num_unique} unique samples")
        else:
            print(f"This chain failed, with {num_unique} unique samples")
        
        end_time = time.time()
        print(f"This chain takes {(end_time - start_time) / 3600} hours")
        if num_chains >= 10:
            break

    theta_all = torch.cat(theta_all, dim=0)
    os.makedirs("BSL_res", exist_ok=True)
    np.save(f"BSL_res/BSL_task{task_id}.npy", theta_all.cpu().numpy())


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
