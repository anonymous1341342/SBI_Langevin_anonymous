# generate training data for both n-model and n-model-5x
from utils_nmodel import *
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
from torch.utils.data import Dataset


# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0

M = 10



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.version.cuda) 
print(torch.cuda.is_available()) 


def main(task_id):
    #############################################################################################
    #        Read previously generated data: data_obs and the SW preconditioned samples         #
    #############################################################################################
    sigma = 0.1 
    obs_size = 1000
    training_size = int(5e5)
    start_time = time.time()


    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()

    ### generate training data
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
    sample_size = training_size 
    theta_r0, data_r0 = gen_ref(mean_theta, std_theta, lower, upper, obs_size, sample_size + 100)

    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask]
    data_r0 = data_r0[~bad_mask]

    print("Inflation rate used:", inf_rate)
    print("Actual inflation rate:", theta_r0.std(dim = 0) / theta_pre.std(dim = 0))
    print(f"number of data points having inf = {bad_mask.sum()}")


    #####################################
    #        Save Training Data         #
    #####################################
    os.makedirs('training_data', exist_ok=True)
    np.save(f'training_data/theta_r0_task{task_id}.npy', theta_r0.cpu().numpy())
    np.save(f'training_data/data_r0_task{task_id}.npy', data_r0.cpu().numpy())

    end_time = time.time()
    print(f"Total time = {round( (end_time - start_time)/60, 2 )} minutes")

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
