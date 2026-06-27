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

# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0

M = 10

def main(task_id):
    training_size = int(1e5) 
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(torch.version.cuda) 
    print(torch.cuda.is_available()) 
    #############################################################################################
    #        Read previously generated data: data_obs and the SW preconditioned samples         #
    #############################################################################################
    sigma = 0.1 
    obs_size = 1000

    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous().to(device)
    x_obs = data_obs[:, 0]
    y_obs = data_obs[:, 1]


    ###############################################
    #          Score matching: round 1            #
    ###############################################

    ### generate training data
    lower = torch.zeros(M + 1) # .to(device)
    lower[0] = a0
    lower[1:] = a
    upper = torch.zeros(M + 1) # .to(device)
    upper[0] = b0
    upper[1:] = b

    sample_size = training_size 

    theta_r0, data_r0 = gen_ref_prior(lower, upper, obs_size, sample_size)
    val_theta_r0, val_data_r0 = gen_ref_prior(lower, upper, obs_size, sample_size)

    print('traing size:', theta_r0.shape[0])

    # calculate the proposal score
    prop_score_r0 = torch.zeros(theta_r0.shape)
    val_prop_score_r0 = torch.zeros(val_theta_r0.shape)


    def g(theta, x):
        d, lower, upper = dist2bd(theta, x)
        return d

    def g1(theta, x):
        d, lower, upper = dist2bd(theta, x)
        return (2 * (theta < (lower + upper) / 2) - 1) # 1 or -1, depends on closer to the lower or upper bound


    ### Train the NN
    # Create DataLoader
    batch_size = int(sys.argv[6]) 
    dataset = TensorDataset(theta_r0, data_r0, prop_score_r0)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    val_dataset = TensorDataset(val_theta_r0, val_data_r0, val_prop_score_r0)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Create model and optimizer
    theta_dim = M + 1 
    x_dim = 2 
    obs_size = data_obs.shape[0] 

    hidden_size = 64 # int(sys.argv[2])  # Number of hidden units
    num_layers = 3 # int(sys.argv[3])    # Number of layers
    num_epochs = 300 # int(sys.argv[4])    # number of epochs
    learning_rate = 1e-3 # float(sys.argv[5])    # learning rate
    print(f"\n hidden_size = {hidden_size}, num_layers = {num_layers}, learning_rate = {learning_rate} \n")


    model = ELU_LikeScoreMatchingNN(theta_dim, x_dim, obs_size, hidden_size, num_layers)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5) # weight_decay = 1e-4

    # train the model
    train_deb(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs = num_epochs)

    # ensure the directory exists
    os.makedirs('prior_model', exist_ok=True)
    torch.save(model, f'prior_model/model_task{task_id}.pth')



    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/3600, 2)} hours')


if __name__ == "__main__":
    task_id = int(sys.argv[1])  
    main(task_id)