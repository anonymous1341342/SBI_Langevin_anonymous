from utils_SI_SSprecond_10floors import *
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
import os
import time
from joblib import Parallel, delayed
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(f"Using device {device}")

def main(task_id):
    start_time = time.time()
    def cal_SS(x):
        """
            sample mean and std
        """
        # sample std fits the normal assumption better than sample var
        return torch.cat((x.mean(dim = 0), x.std(dim = 0)), dim=0)
        # return torch.cat((x.mean(dim = 0), x.var(dim = 0)), dim=0)

    def BSL_SI_10F(data_obs, log_theta_init, prop_scale, n_simu, maxiter):
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
        
        import sys
        try:
            sys.stdout.reconfigure(line_buffering=True)  # py>=3.7
        except Exception:
            pass


        
        data_obs = data_obs.to(device)
        data_obs = get_SS(data_obs).reshape(-1, 52).T
        data_obs = data_obs[:, 1:] # drop the first column to avoid linear dependence
        SS_obs = cal_SS( data_obs ) # sample mean and std

        log_theta0 = log_theta_init.clone().to(device)
        log_theta_path = [log_theta0]

        # calculate the likelihood at log_theta0
        # First: obtain the Summary statistics over n_simu simulated samples
        SS_set0 = torch.zeros(n_simu, 22).to(device)
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
            t1 = time.time()
            
            log_theta_prop = log_theta0 + prop_scale * torch.randn(log_theta0.shape).to(device) # propose
            # print(f"log_theta_prop = {log_theta_prop}")
                    
            # calculate the likelihood at theta_prop
            SS_set_prop = torch.zeros(n_simu, 22).to(device)
            for j in range(n_simu):
                # generate simulated data
                z = gen_z(N, T)
                data = m_vec_partial(N, T, log_theta_prop.exp(), gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
                data = get_SS(data).reshape(-1, 52).T
                data = data[:, 1:] # drop the first column to avoid exact linear dependence
                SS_set_prop[j] = cal_SS(data)
            mu_prop = SS_set_prop.mean(dim = 0)
            Cov_prop = 1/(n_simu-1) * ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) ).T @ ( SS_set_prop - mu_prop.view(1, -1).repeat(n_simu, 1) )

            acc_prob = torch.linalg.det(Cov0.to(torch.float64))**(1/2) / torch.linalg.det(Cov_prop.to(torch.float64))**(1/2) * torch.exp(
                -0.5 * (SS_obs - mu_prop) @ torch.linalg.solve(Cov_prop, SS_obs - mu_prop) + 0.5 * (SS_obs - mu0) @ torch.linalg.solve(Cov0, SS_obs - mu0))
            acc_prob = acc_prob * (-1/8 * torch.linalg.norm(log_theta_prop + 3.)**2 + 1/8 * torch.linalg.norm(log_theta0 + 3.)**2).exp() # the prior is not uniform

            if torch.rand(1) <= acc_prob.cpu(): # accept
                # print("accept")
                log_theta_path.append(log_theta_prop.clone())
                log_theta0 = log_theta_prop.clone()
                mu0 = mu_prop
                Cov0 = Cov_prop
            else: # do not accept
                # print("reject")
                log_theta_path.append(log_theta0.clone())
            t2 = time.time()
            print(f"Iteration {iter+1}/{maxiter}, time: {round(t2 - t1, 3)} seconds", flush=True)
        return log_theta_path    



    # Settings for the floor and room assignments
    K = 10 # number of floors
    N = 600
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



    pre_samples = pd.read_csv(f"res_precond/pre_samples_lam{0}_task{task_id}.csv")
    pre_samples = torch.tensor(pre_samples.values, dtype = torch.float32).contiguous().to(device)
    mean_theta = pre_samples.mean(dim = 0)
    std_theta = pre_samples.std(dim = 0)

    ### Reference Table
    sample_size = int(10000)
    mu_new = mean_theta.view(1, -1).repeat(sample_size, 1)
    sigma_new = std_theta.view(1, -1).repeat(sample_size, 1)

    log_theta = mu_new + sigma_new * torch.randn(mu_new.shape).to(device)
    log_theta = log_theta.to(device)
    log_theta_set = log_theta

    ### Observed data
    data_obs = pd.read_csv(f"data_obs/y_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous()



    def get_one_chain():
        import sys
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
        
        log_theta_init = log_theta_set[np.random.randint(0, log_theta_set.shape[0])]
        prop_scale = 0.1 * log_theta_set.std(dim = 0)
        n_simu = 200 
        maxiter = 1000 
        
        theta_path = BSL_SI_10F(data_obs, log_theta_init, prop_scale, n_simu, maxiter)
        theta_path = torch.stack(theta_path, dim=0)
        return theta_path[201:].cpu() # 200 burn-in



    n_alloc = int(os.environ.get("SLURM_CPUS_ON_NODE", "11"))
    n_jobs = max(1, n_alloc - 1)
    print(f"Using {n_jobs}/{n_alloc} CPUs for sampling.", flush=True)
    
    all_chains = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(get_one_chain)() for _ in range(10)
    )
    
    all_chains = torch.cat(all_chains, dim=0)
    # ensure the directory exists
    os.makedirs("BSL_res", exist_ok=True)
    np.save(f"BSL_res/BSL_task_{task_id}.npy", all_chains.cpu().numpy())

    end_time = time.time()
    print(f"Total time = {round( (end_time - start_time)/3600, 3)} hours")



if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
