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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def ABC_W1(data_obs, log_theta_set):
    """
    ABC by comparing W1(data_obs, data_simu)
    """
    # data_obs is actually SS_obs
    data_obs = data_obs.to(device)
    log_theta_set = log_theta_set.to(device)
    W1_set = torch.zeros(log_theta_set.shape[0])
    
    for i in range(log_theta_set.shape[0]):
        # generate simulated data based on theta_i
        theta = log_theta_set[i].exp()
        z = gen_z32(N, T) # latent variable
        data_simu = m_vec_partial32(N, T, theta, gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
        data_simu = get_SS(data_simu).reshape(-1, 52).T # the summary statistics
    
        # calculate W1(data_obs, data_simu)
        marg1 = (1/data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device) # marginal distribution
        marg2 = (1/data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device)
        cost_mat = ot.dist(data_obs, data_simu, metric='euclidean')  
        W1_set[i] = ot.emd2(marg1, marg2, cost_mat)
    return W1_set


def main(task_id):
    pre_samples = pd.read_csv(f"res_precond/pre_samples_lam{0}_task{task_id}.csv")
    pre_samples = torch.tensor(pre_samples.values, dtype = torch.float32).contiguous().to(device)
    mean_theta = pre_samples.mean(dim = 0)
    std_theta = pre_samples.std(dim = 0)

    ### Reference Table
    sample_size = 20000 # int(sys.argv[2])
    mu_new = mean_theta.view(1, -1).repeat(sample_size, 1)
    sigma_new = std_theta.view(1, -1).repeat(sample_size, 1)

    log_theta = mu_new + sigma_new * torch.randn(mu_new.shape).to(device)
    log_theta = log_theta.to(device)
    log_theta_set = log_theta

    ### Observed data
    data_obs = pd.read_csv(f"data_obs/y_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous()
    data_obs = get_SS(data_obs).reshape(-1, 52).T # transform to Summary Statistics


    ### Run ABC_W1
    print("Running ABC with W1 loss")
    W1_set = ABC_W1(data_obs, log_theta_set)
    smallest_values, smallest_indices = torch.topk(W1_set, 100, largest=False)
    log_theta_r1 = log_theta_set[smallest_indices].clone()

    # ensure the directory exists
    os.makedirs("ABC_res", exist_ok=True)

    pd.DataFrame( log_theta_r1.cpu().numpy() ).to_csv(f'ABC_res/log_theta_r1_ABCW1_task{task_id}.csv', index=False, header=False)


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
