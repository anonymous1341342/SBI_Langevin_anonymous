from utils_npe import *
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import math
import time
# from tqdm import tqdm
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


training_size = 10000 # int(float(sys.argv[2]))
batch_size = 5000 # int(float(sys.argv[3]))
print(f"training_size = {training_size}, batch_size = {batch_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.version.cuda) 
print(torch.cuda.is_available()) 


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

    F_assign = F_assign.to(device)
    C_F = C_F.to(device)
    C_R = C_R.to(device)

    gamma = 0.05
    alpha = 0.1
    eta = 0.1 
    T = 52

    start_time = time.time()
    #############################################################################################
    #        Read previously generated data: data_obs and the SW preconditioned samples         #
    #############################################################################################
    pre_samples = pd.read_csv(f"res_precond/pre_samples_lam{0}_task{task_id}.csv")
    pre_samples = torch.tensor(pre_samples.values, dtype = torch.float32).contiguous().to(device)


    ##########################################
    #          Generate training data        #
    ##########################################
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

    ###############################
    #          NN training        #
    ###############################
    batch_size = batch_size # data_r0.shape[0]
    dataset = TensorDataset(data_r0, log_theta_r0)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    val_dataset = TensorDataset(val_data_r0, val_log_theta_r0)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)


    model = GaussianDensityNetwork(d_x=data_r0.shape[1], d_theta=log_theta_r0.shape[1],
                                d_model=32, lr=0.001, weight_decay=0.0, mean_field=False)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',     # name of the logged metric to monitor
        patience=50,            # number of epochs with no improvement after which training will stop
        mode='min',             # minimize the monitored metric
        verbose=False            # print a message when stopping
    )

    trainer = L.Trainer(
        max_epochs=5000, # 1000,
        callbacks=[early_stop_callback],
        accelerator='gpu', 
        devices=1,
    )


    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)

    # ensure the directory exists
    os.makedirs("logNPE_model", exist_ok=True)
    torch.save(model, f'logNPE_model/model_task{task_id}.pth')

    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/60, 2)} minutes')



if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)


