from utils_SI_SSprecond_10floors import *
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math
import time
# from tqdm import tqdm
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main(task_id):
    start_time = time.time()


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

    F_assign = F_assign.to(torch.float64).to(device)
    C_F = C_F.to(torch.float64).to(device)
    C_R = C_R.to(torch.float64).to(device)

    gamma = 0.05
    alpha = 0.1
    eta = 0.1 
    T = 52

    #############################################
    #        Read the observed data         #
    #############################################
    y_obs = pd.read_csv(f"data_obs/y_obs_task{task_id}.csv")
    y_obs = torch.tensor(y_obs.values, dtype = torch.float64).contiguous().to(device)
    SS_obs = get_SS(y_obs).reshape(-1, 52).T

    ###################################################
    #        Preconditioning by minimizing SW         #
    ###################################################
    metric = "W1"
    simu_size = y_obs.shape[0]
    num_diffz = 100
    lr = 0.1
    maxiter = 100 

    for lam_time in [0]:
        pre_samples = cm_SS_log_res_diffz(lam_time, metric, SS_obs, simu_size, num_diffz, lr, maxiter)
        df_pre_samples = pd.DataFrame(pre_samples.cpu())
        # ensure the directory exists
        os.makedirs("res_precond", exist_ok=True)
        df_pre_samples.to_csv(f"res_precond/pre_samples_lam{lam_time}_task{task_id}.csv", index=False)


    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/60, 2)} minutes')


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
