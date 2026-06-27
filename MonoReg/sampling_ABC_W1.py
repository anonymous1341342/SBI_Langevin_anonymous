from utils_monoBP_single import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
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

    x_obs = data_obs[:, 0]
    y_obs = data_obs[:, 1]



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

    # one theta with one x
    sample_size = int(1e6) 
    print(f"Using sample size {sample_size:.1e}")
    theta_r0, data_r0 = gen_ref_distinct_theta(mean_theta, std_theta, lower, upper, sample_size + 100)

    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask]
    data_r0 = data_r0[~bad_mask]



    t1 = time.time()
    ####################
    #   ABC with W1   #
    ####################
    def ABC_W1(data_obs, theta_set):
        """
        ABC by comparing W1(data_obs, data_simu)
        """
        data_obs = data_obs.to(device)
        theta_set = theta_set.to(device)
        W1_set = torch.zeros(theta_set.shape[0])
        
        for i in range(theta_set.shape[0]):
            # generate simulated data based on theta_i
            theta = theta_set[i]
            x = torch.rand(obs_size).to(device)
            A = get_A(M).to(device)
            psi = get_psi(x, M).to(device)
            y = psi @ torch.linalg.inv(A) @ theta + sigma * torch.randn(obs_size).to(device)
            data_simu = torch.stack((x, y), dim = 1)
        
            # calculate W1(data_obs, data_simu)
            marg1 = (1/data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device) # marginal distribution
            marg2 = (1/data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device)
            cost_mat = ot.dist(data_obs.to(device), data_simu, metric='euclidean')  
            W1_set[i] = ot.emd2(marg1, marg2, cost_mat)
            # W1_set[i] = ot.sinkhorn2(marg1, marg2, cost_mat, 0.1) # entropy regularized, not faster in this case (n = 1000)
        return W1_set

    theta_set = theta_r0
    W1_set = ABC_W1(data_obs, theta_set)
    smallest_values, smallest_indices = torch.topk(W1_set, 1000, largest=False) # top 1000 before
    theta_r1 = theta_set[smallest_indices].clone()


    A = get_A(M)
    grids = torch.arange(0, 1.01, 0.01)
    psi_grids = get_psi(grids, M)
    pred_ys_ABCW1 = psi_grids @ torch.linalg.inv(A) @ theta_r1.T

    # ensure the directory exists
    os.makedirs('sample_res_others', exist_ok=True)

    pd.DataFrame( pred_ys_ABCW1.numpy() ).to_csv(f'sample_res_others/pred_ys_ABCW1_task{task_id}.csv', index=False, header=False)
    np.save(f'sample_res_others/theta_r1_ABCW1_task{task_id}.npy', theta_r1.cpu().numpy())


    t2 = time.time()
    print(f"ABCW1 used {(t2-t1)/60} mins")


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)

