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
    #############################################
    #        Generate the observed data         #
    #############################################
    gamma = 0.05
    alpha = 0.1
    eta = 0.1 
    T = 52

    beta_true = torch.tensor([0.05, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.05], dtype = torch.float64).to(device)

    z_obs = gen_z(N, T)
    y_obs = m_vec_partial(N, T, beta_true, gamma, alpha, eta, F_assign, C_F, C_R, z_obs, NF, NR)

    df_y_obs = pd.DataFrame(y_obs.cpu())
    os.makedirs("data_obs", exist_ok=True)
    df_y_obs.to_csv(f"data_obs/y_obs_task{task_id}.csv", index=False)



if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
