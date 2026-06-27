# Check loss: single training, n training, n posterior
from utils_nmodel import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import sys
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma = 0.1 # noise level
obs_size = 1000
# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0


# config = json.loads(sys.argv[2])
# Fisher = (config["Fisher"] == "True")
# training_size = int(config["training_size"])

def main(task_id, Fisher, training_size):
    # SW samples for generating training data
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

    start_time = time.time()
    # ================== Load the model ================ #
    if Fisher:
        model = ELU_LikeScoreMatchingNN(theta_dim = M + 1, x_dim = 2, obs_size = 1000, hidden_size= 64, num_layers = 3).to(device)
        checkpoint = torch.load(f'nmodel_fisher/checkpoint_task{task_id}_trainsize{training_size}.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        bias_lastlayer = checkpoint['bias_lastlayer']
        
        with torch.no_grad(): 
            model.layers[-1].bias -= bias_lastlayer / obs_size
        
        def NScore(theta, x): # the input x is "N-data"
            return model(theta, x)

        def SingleScore(theta, x):
            return model.layers(torch.cat((theta, x), dim = 1))

    else:
        model = ELU_LikeScoreMatchingNN(theta_dim = M + 1, x_dim = 2, obs_size = 1000, hidden_size= 64, num_layers = 3).to(device)
        checkpoint = torch.load(f'nmodel_init/checkpoint_task{task_id}_trainsize{training_size}.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        bias_lastlayer = checkpoint['bias_lastlayer']
        
        with torch.no_grad(): 
            model.layers[-1].bias -= bias_lastlayer / obs_size
        
        def NScore(theta, x): # the input x is "N-data"
            return model(theta, x)

        def SingleScore(theta, x):
            return model.layers(torch.cat((theta, x), dim = 1))


    res = {'sm_single': None, 'sm_n': None, 'sm_n_post': None}
    scale = {'sm_single': None, 'sm_n': None, 'sm_n_post': None}
    # ================== sm loss on single data, training distribution ================ #
    sample_size = 100000
    theta_r0, data_r0 = gen_ref_distinct_theta(mean_theta, std_theta, lower, upper, sample_size)
    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask]
    data_r0 = data_r0[~bad_mask]

    x_r0 = data_r0[:, ::2]
    y_r0 = data_r0[:, 1::2]

    true_score = torch.zeros(sample_size, M + 1)
    for i in range(sample_size):
        psi = get_psi(x_r0[i], M)
        A = get_A(M)
        design = psi @ torch.linalg.inv(A)
        true_score[i] = 1/sigma**2 * (design.T @ y_r0[i] - design.T @ design @ theta_r0[i])

    # Estimated score
    bsize = 1000
    est_single = torch.zeros(sample_size, M + 1)
    for i in range(int(sample_size / bsize)):
        est_single[(i*bsize):((i+1)*bsize)] = SingleScore(theta_r0[(i*bsize):((i+1)*bsize)].to(device), data_r0[(i*bsize):((i+1)*bsize)].to(device)).detach().cpu()
        
    res['sm_single'] = 0.5 * ( (est_single - true_score)**2 ).sum(dim = 1).mean().item()
    scale['sm_single'] = 0.5 * ( (true_score)**2 ).sum(dim = 1).mean().item()

    print("Finished sm_single")
    # ================== sm loss on n data, training distribution ================ #
    sample_size = 100000
    theta_r0, data_r0 = gen_ref(mean_theta, std_theta, lower, upper, obs_size, sample_size)

    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask]
    data_r0 = data_r0[~bad_mask]

    x_r0 = data_r0[:, ::2]
    y_r0 = data_r0[:, 1::2]

    true_score = torch.zeros(sample_size, M + 1)
    for i in range(sample_size):
        psi = get_psi(x_r0[i], M)
        A = get_A(M)
        design = psi @ torch.linalg.inv(A)
        true_score[i] = 1/sigma**2 * (design.T @ y_r0[i] - design.T @ design @ theta_r0[i])

    # Estimated score
    bsize = 1000
    est_n = torch.zeros(sample_size, M + 1)
    for i in range(int(sample_size / bsize)):
        est_n[(i*bsize):((i+1)*bsize)] = NScore(theta_r0[(i*bsize):((i+1)*bsize)].to(device), data_r0[(i*bsize):((i+1)*bsize)].to(device)).detach().cpu()
        
    res['sm_n'] = 0.5 * ( (est_n - true_score)**2 ).sum(dim = 1).mean().item()
    scale['sm_n'] = 0.5 * ( (true_score)**2 ).sum(dim = 1).mean().item()

    print("Finished sm_n")
    # ================== sm loss on n data, posterior distribution ================ #
    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous().to(device)

    samples_gibbs = torch.tensor(np.load(f'res_gibbs/theta_gibbs_task{task_id}.npy'), dtype=torch.float32)


    x_obs = data_obs[:, 0].cpu()
    y_obs = data_obs[:, 1].cpu()

    sample_size = 50000
    theta_post = samples_gibbs.to(device)[:sample_size].clone()

    true_score = torch.zeros(sample_size, M + 1)
    for i in range(sample_size):
        psi = get_psi(x_obs, M)
        A = get_A(M)
        design = psi @ torch.linalg.inv(A)
        true_score[i] = 1/sigma**2 * (design.T @ y_obs - design.T @ design @ theta_post[i].cpu())

    # Estimated score
    bsize = 1000
    est_n_post = torch.zeros(sample_size, M + 1)
    for i in range(int(sample_size / bsize)):
        est_n_post[(i*bsize):((i+1)*bsize)] = NScore(theta_post[(i*bsize):((i+1)*bsize)].to(device), data_obs.view(1, -1).repeat(bsize, 1).to(device)).detach().cpu()
        
    res['sm_n_post'] = 0.5 * ( (est_n_post - true_score)**2 ).sum(dim = 1).mean().item()
    scale['sm_n_post'] = 0.5 * ( (true_score)**2 ).sum(dim = 1).mean().item()

    print("Finished sm_n_post")


    # ======== Save the results ======== #
    if Fisher:
        os.makedirs("check_loss", exist_ok=True)
        pd.DataFrame([res]).to_csv(f'check_loss/loss_task{task_id}_trainsize{training_size}_fisher.csv', index=False)
        pd.DataFrame([scale]).to_csv(f'check_loss/scale_task{task_id}_trainsize{training_size}_fisher.csv', index=False)
    else:
        os.makedirs("check_loss", exist_ok=True)
        pd.DataFrame([res]).to_csv(f'check_loss/loss_task{task_id}_trainsize{training_size}_init.csv', index=False)
        pd.DataFrame([scale]).to_csv(f'check_loss/scale_task{task_id}_trainsize{training_size}_init.csv', index=False)

    # ======== Record the total time ======== #
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/60, 2)} minutes')



if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id, Fisher = False, training_size = int(1e5))
    main(task_id, Fisher = True, training_size = int(1e5))
    main(task_id, Fisher = True, training_size = int(5e5))
