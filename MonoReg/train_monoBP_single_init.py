from utils_monoBP_single import *
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
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0

M = 10


hidden_size = 64 # config["hidden_size"]
num_layers = 3 # config["num_layers"]
num_epochs = 100 # config["num_epochs"]
early_stop_patience = 15 # config["early_stop_patience"]
learning_rate = 1e-3 # config["learning_rate"]
batch_size = int(1e3) # int(config["batch_size"])
training_size = int(1e6) # int(config["training_size"])
sched = True
sched_patience = 5 # int(config["sched_patience"])




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.version.cuda) 
print(torch.cuda.is_available()) 

def main(task_id):
    start_time = time.time()
    #############################################################################################
    #        Read previously generated data: data_obs and the SW preconditioned samples         #
    #############################################################################################
    sigma = 0.1 
    obs_size = 1000


    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()

    ###############################################
    #          Score matching: round 1            #
    ###############################################

    # proposal distribution given by the preconditioning samples
    # Determine the proposal distribution and then generate the reference table for training
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

    sample_size = training_size 
    ### Read the same training data
    theta_r0 = np.load(f'single_training_data/theta_r0_task{task_id}.npy', mmap_mode='r')[:sample_size]
    data_r0  = np.load(f'single_training_data/data_r0_task{task_id}.npy',  mmap_mode='r')[:sample_size]
    theta_r0 = torch.tensor(theta_r0, dtype = torch.float32).contiguous()
    data_r0 = torch.tensor(data_r0, dtype = torch.float32).contiguous()

    # Generate the validation data
    val_theta_r0, val_data_r0 = gen_ref_distinct_theta(mean_theta, std_theta, lower, upper, int(1e6) + 100)

    bad_mask = torch.isinf(val_theta_r0).any(dim=1)
    val_theta_r0 = val_theta_r0[~bad_mask]
    val_data_r0 = val_data_r0[~bad_mask]

    val_theta_r0 = val_theta_r0[:int(1e6)]
    val_data_r0 = val_data_r0[:int(1e6)]

    print('traing size:', theta_r0.shape[0])
    print('validation size:', val_theta_r0.shape[0])

    # calculate the proposal score
    prop_score_r0 = ( mean_theta.repeat(theta_r0.shape[0], 1) - theta_r0 ) / (std_theta**2).repeat(theta_r0.shape[0], 1)
    val_prop_score_r0 = ( mean_theta.repeat(val_theta_r0.shape[0], 1) - val_theta_r0 ) / (std_theta**2).repeat(val_theta_r0.shape[0], 1)


    #####################################################
    #          Determine the weight function            #
    #####################################################
    # here we make sure the scale of the weight function is the same for all dimensions
    # dist = dist2bd(theta_r0.to(device), data_r0.to(device))[0]

    scale = dist2bd(theta_r0.to(device), data_r0.to(device))[0].mean(dim = 0, keepdim = True)

    def make_g_functions(scale):
        def g(theta, x):
            d, lower, upper = dist2bd(theta, x)
            return d / scale

        def g1(theta, x):
            d, lower, upper = dist2bd(theta, x)
            return (2 * (theta < (lower + upper) / 2) - 1) / scale # 1 or -1, depends on closer to the lower or upper bound
        
        return g, g1

    g, g1 = make_g_functions(scale)




    #######################################
    #          Training starts            #
    #######################################

    ### Train the NN
    # Create DataLoader

    dataset = TensorDataset(theta_r0, data_r0, prop_score_r0)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    val_dataset = TensorDataset(val_theta_r0, val_data_r0, val_prop_score_r0)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Create model and optimizer
    theta_dim = M + 1 
    x_dim = 2 
    print(f"\n hidden_size = {hidden_size}, num_layers = {num_layers}, learning_rate = {learning_rate} \n")

    model = single_ELU_LikeScoreMatchingNN(theta_dim, x_dim, obs_size, hidden_size, num_layers)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

    scheduler = None
    if sched:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)
        print(f"Scheduler is used, with patience {sched_patience}")

    # train the model
    bias_lastlayer = train_deb(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bias_lastlayer': bias_lastlayer
    }, f'model_single_init/checkpoint_task{task_id}_trainsize{training_size}.pth')



    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/3600, 2)} hours')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)
