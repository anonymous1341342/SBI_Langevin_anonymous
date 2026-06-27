from utils_SW1Localization import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import sys
import json


# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0

M = 10


def res_diffz(data_obs, simu_size, num_diffz):
    theta_res_diffz = torch.zeros(num_diffz, M + 1)
    alpha_res_diffz = torch.zeros(num_diffz, M + 1)
    
    u_size = 100
    lr = 0.1 
    maxiter = 1000 
    for i in range(num_diffz):
        print(f"\nRunning {i+1}/{num_diffz}")
        # generate z
        z = gen_z(simu_size)
        
        # random initialization
        theta_init = torch.rand(M + 1)
        theta_init[0] = torch.rand(1)
    
        # solve the SW objective by Adam
        theta_path, alpha_path, train_loss_path = Adam_SW1_fixz_repar(data_obs, u_size, z, theta_init, lr, maxiter, scheduler_patience = 30, early_stop_patience = 70)
        
        # record the solution
        theta_res_diffz[i] = theta_path[np.argmin(train_loss_path)]
        alpha_res_diffz[i] = alpha_path[np.argmin(train_loss_path)]
    return theta_res_diffz, alpha_res_diffz





def main(task_id):
    start_time = time.time()

    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype=torch.float32).contiguous()

    simu_size = data_obs.shape[0]
    num_diffz = 100
    theta_res_diffz, alpha_res_diffz = res_diffz(data_obs, simu_size, num_diffz)

    # ensure the directory exists
    os.makedirs("res_SW1_precond", exist_ok=True)

    pd.DataFrame(theta_res_diffz.cpu().numpy()).to_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv", index=False)
    pd.DataFrame(alpha_res_diffz.cpu().numpy()).to_csv(f"res_SW1_precond/alpha_pre_task{task_id}.csv", index=False)

    end_time = time.time()
    print(f"Total time = {(end_time-start_time)/60:.2f} minutes")


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)