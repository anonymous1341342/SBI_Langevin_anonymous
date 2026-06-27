# no truncation for the proposal distribution
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
from pathlib import Path

# prior for theta0
a0 = -float('inf') # -5.0
b0 = float('inf')  # 5.0
# prior for theta1-thetaM
a = -float('inf') # 0.0
b = float('inf') # 1.0

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


def main(task_id):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(torch.version.cuda) 
    print(torch.cuda.is_available()) 

    def sample_trunc_fast(mean_theta, prop_cov, lower, upper, sample_size, batch_size=10000):
        dist = torch.distributions.MultivariateNormal(
            loc=mean_theta,
            covariance_matrix=prop_cov
        )

        dim = lower.numel()
        out = torch.empty(sample_size, dim, device=lower.device, dtype=lower.dtype)

        filled = 0

        while filled < sample_size:
            theta = dist.sample((batch_size,))
            mask = ((theta >= lower) & (theta <= upper)).all(dim=1)
            valid = theta[mask]

            n = min(valid.shape[0], sample_size - filled)
            if n > 0:
                out[filled:filled+n] = valid[:n]
                filled += n

        return out

    def gen_ref_distinct_theta_nondiag(prop_mean, prop_cov, lower, upper, sample_size = 10000):
        """
            generate theta from a (truncated) gaussian proposal distribution, and then use theta to generate x
            mean_theta: the mean of the truncated normal, a 1-dim tensor of the same length as theta
            std_theta: the std of the truncated normal, a 1-dim tensor of the same length as theta
            lower: the lower bound for each dimension of theta, a 1-dim tensor
            upper: the upper bound for each dimension of theta, a 1-dim tensor
        """
        time_start = time.time()
        
        theta_time1 = time.time()
        theta = sample_trunc_fast(prop_mean, prop_cov, lower, upper, sample_size)
        theta_time2 = time.time()
        print(f"time of generating theta: {round(theta_time2 - theta_time1)} seconds")


        x = torch.rand(sample_size)
        A = get_A(M)
        psi = get_psi(x, M) # (sample_size, M + 1)
        y = ( (psi @ torch.linalg.inv(A)) * theta ).sum(dim = 1) + sigma * torch.randn(sample_size)
        
        
        data = torch.zeros(sample_size, 2)
        data[:, 0] = x
        data[:, 1] = y

        time_end = time.time()
        print(f"Total time for generating ABC table = {round((time_end - time_start) / 60, 3)} minutes")
        return theta, data


    #############################################################################################
    #        Read previously generated data: data_obs and the SW preconditioned samples         #
    #############################################################################################
    sigma = 0.1 
    obs_size = 1000

    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous().to(device)


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


    inf_rate = actual_inf_rate

    prop_mean = theta_pre.mean(dim = 0)
    prop_cov = torch.diag(inf_rate) @ torch.cov(theta_pre.T) @ torch.diag(inf_rate)

    sample_size = training_size 

    # ======= ref_S ====== #
    path_theta = Path(f'ref_S/theta_r0_task{task_id}.npy')
    path_data = Path(f'ref_S/data_r0_task{task_id}.npy')
    path_val_theta = Path(f'ref_S/val_theta_r0_task{task_id}.npy')
    path_val_data = Path(f'ref_S/val_data_r0_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists

    if not path_theta.exists():
        theta_r0, data_r0 = gen_ref_distinct_theta_nondiag(prop_mean, prop_cov, lower, upper, sample_size + 100)
        val_theta_r0, val_data_r0 = gen_ref_distinct_theta_nondiag(prop_mean, prop_cov, lower, upper, sample_size + 100)
        np.save(path_theta, theta_r0.numpy())
        np.save(path_data, data_r0.numpy())
        np.save(path_val_theta, val_theta_r0.numpy())
        np.save(path_val_data, val_data_r0.numpy())
    else:
        theta_r0 = torch.from_numpy(np.load(path_theta))
        data_r0 = torch.from_numpy(np.load(path_data))
        val_theta_r0 = torch.from_numpy(np.load(path_val_theta))
        val_data_r0 = torch.from_numpy(np.load(path_val_data))

    # remove nan for training data
    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask]
    data_r0 = data_r0[~bad_mask]

    theta_r0 = theta_r0[:sample_size]
    data_r0 = data_r0[:sample_size]


    # remove nan for validation data
    bad_mask = torch.isinf(val_theta_r0).any(dim=1)
    val_theta_r0 = val_theta_r0[~bad_mask]
    val_data_r0 = val_data_r0[~bad_mask]

    val_theta_r0 = val_theta_r0[:sample_size]
    val_data_r0 = val_data_r0[:sample_size]

    print('training size:', theta_r0.shape[0])
    print('validation size:', val_theta_r0.shape[0])

    prop_score_r0 = torch.linalg.solve(prop_cov, (prop_mean - theta_r0).T).T
    val_prop_score_r0 = torch.linalg.solve(prop_cov, (prop_mean - val_theta_r0).T).T

    #####################################################
    #          Determine the weight function            #
    #####################################################
    # here we make sure the scale of the weight function is the same for all dimensions
    # dist = dist2bd(theta_r0.to(device), data_r0.to(device))[0]

    scale = dist2bd(theta_r0.to(device), data_r0.to(device))[0].mean(dim = 0, keepdim = True)

    def make_g_functions(scale):
        def g(theta, x):
            return torch.ones_like(theta)

        def g1(theta, x):
            return torch.zeros_like(theta)
        
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
    theta_dim = M + 1 # 3
    x_dim = 2 # 5
    obs_size = data_obs.shape[0] # 500
    print(f"\n hidden_size = {hidden_size}, num_layers = {num_layers}, learning_rate = {learning_rate} \n")

    model = single_ELU_LikeScoreMatchingNN(theta_dim, x_dim, obs_size, hidden_size, num_layers)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

    scheduler = None
    if sched:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)
        print(f"Scheduler is used, with patience {sched_patience}")

    # train the model
    bias_lastlayer = train_deb(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience)


    # ensure the folder exists
    os.makedirs('model_single_init', exist_ok=True)

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
    task_id = int(sys.argv[1])
    main(task_id)

