# do not inflate the std when building the proposal distribution
from source_SI10F_scomat import *
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
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

task_id = sys.argv[1]
config = json.loads(sys.argv[2])
training_size = config["training_size"]
nnsize = config["nnsize"]
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
weight_decay = config["weight_decay"]

sched = config["sched"] == "True"
early_stop_patience = config["early_stop_patience"]




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.version.cuda) 
print(torch.cuda.is_available()) 

# no prior boundary, we use log_theta

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

gamma = 0.05
alpha = 0.1
eta = 0.1 # 0.5
T = 52

start_time = time.time()
#############################################################################################
#        Read previously generated data: data_obs and the SW preconditioned samples         #
#############################################################################################
pre_samples = pd.read_csv(f"res_precond/pre_samples_lam{0}_task{task_id}.csv")
pre_samples = torch.tensor(pre_samples.values, dtype = torch.float32).contiguous().to(device)

##################################
#          Score matching        #
##################################

# proposal distribution given by the preconditioning samples
# Determine the proposal distribution and then generate the reference table for training
mean_theta = pre_samples.mean(dim = 0)
std_theta = pre_samples.std(dim = 0)


sample_size = training_size # 10000
t1 = time.time()
log_theta_r0, data_r0 = gen_ref_log(mean_theta, std_theta, sample_size)
val_log_theta_r0, val_data_r0 = gen_ref_log(mean_theta, std_theta, sample_size)
t2 = time.time()
print(f'Time of generating training and validation data: {round( (t2-t1)/60, 2 )} minutes')

print('traing size:', log_theta_r0.shape[0])


# calculate the proposal score
prop_score_r0 = ( mean_theta.repeat(log_theta_r0.shape[0], 1) - log_theta_r0 ) / (std_theta**2).repeat(log_theta_r0.shape[0], 1)
val_prop_score_r0 = ( mean_theta.repeat(val_log_theta_r0.shape[0], 1) - val_log_theta_r0 ) / (std_theta**2).repeat(val_log_theta_r0.shape[0], 1)

### Train the NN
# Create DataLoader
dataset = TensorDataset(log_theta_r0, data_r0, prop_score_r0)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
val_dataset = TensorDataset(val_log_theta_r0, val_data_r0, val_prop_score_r0)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

# Create model and optimizer
input_size = (K+2) + (K+2)*T
output_size = K+2

print(f"\n num_epochs = {num_epochs}, learning_rate = {learning_rate} \n")


if nnsize == "medium":
    print("Using Medium ELU nn")
    model = ELU_Nonadd_Medium(input_size, output_size)
if nnsize == "small":
    print("Using small ELU nn")
    model = ELU_Nonadd_Small(input_size, output_size)
if nnsize == "large":
    print("Using large ELU nn")
    model = ELU_Nonadd_Large(input_size, output_size)
if nnsize == "Xsmall":
    print("Using Xsmall ELU nn")
    model = ELU_Nonadd_XSmall(input_size, output_size)


optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay) # weight_decay = 1e-5

scheduler = None
if sched:
    sched_patience = int(config["sched_patience"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)
    print(f"Scheduler is used, with patience {sched_patience}")

# train the model
bias_lastlayer = train_deb5(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience)

os.makedirs("model_10F", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'bias_lastlayer': bias_lastlayer
}, f'model_10F/checkpoint_task{task_id}.pth')

#############################################
#          Record the total time            #
#############################################
end_time = time.time()
total_duration = end_time - start_time
print(f'Total time: {round(total_duration/3600, 2)} hours')

