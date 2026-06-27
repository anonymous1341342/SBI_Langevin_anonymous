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





sigma = 0.1 # noise level
obs_size = 1000
# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0




def main(task_id):
    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous()

    x_obs = data_obs[:, 0]
    y_obs = data_obs[:, 1]

    psi = get_psi(x_obs, M)
    A = get_A(M)
    # OLS
    design = psi @ torch.linalg.inv(A)
    OLS = torch.linalg.solve(design.T @ design, design.T @ y_obs)

    # ================== Load the Deb_SingleModel and its DebReg model ================ #
    model = ELU_LikeScoreMatchingNN(theta_dim = M + 1, x_dim = 2, obs_size = 1000, hidden_size= 64, num_layers = 3).to(device)
    checkpoint = torch.load(f'nmodel_fisher_nondiag/checkpoint_task{task_id}_trainsize100000.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    bias_lastlayer = checkpoint['bias_lastlayer']

    with torch.no_grad(): 
        model.layers[-1].bias -= bias_lastlayer / obs_size

    def NScore_DebReged(theta, x): # the input x is "N-data"
        return model(theta, x)


    t1 = time.time()
    #######################################
    #               Annealing             #
    #######################################
    def repar_draw_post_vec_annealing3(skip, annealing_set, lower, upper, x_obs, theta_init, epis = 0.001, S = 100):
        # skip: in the last run, keep 1 sample for every $skip$ samples
        # vectorized, generate multiple MC chains, but only return the last draws of each chain
        # epis: step size
        # S: length of each chain
        # theta_init: dim m*d
        
        theta0 = theta_init.view(-1, M + 1).to(device) # initial value of theta
        alpha0 = ( (theta0 - lower) / (upper - theta0) ).log()
        alpha0[:, 0] = theta0[:, 0]
        x_obs = x_obs.to(device).view(1, -1)

        res = []
        for id_ann in range(annealing_set.shape[0]):
            ann_par = annealing_set[id_ann]
            print(f"annealing parameter: {ann_par}")

            for i in range(S):
                prior_score = (1 - alpha0.exp()) / (1 + alpha0.exp())
                prior_score[:, 0] = 0
                like_score_theta = NScore_DebReged(theta0, x_obs.repeat(theta0.shape[0], 1)).detach().to(device)
                like_score = ( alpha0.exp() * (upper - lower) / (1 + alpha0.exp())**2 ) * like_score_theta
                like_score[:, 0] = like_score_theta[:, 0]
                alpha1 = alpha0 + epis * (like_score + prior_score) * ann_par + np.sqrt(2.0 * epis) * torch.randn(alpha0.shape).to(device)
                theta1 = (lower + upper * alpha1.exp()) / (1 + alpha1.exp())
                theta1[:, 0] = alpha1[:, 0]
                
                theta0 = theta1
                alpha0 = alpha1

                # store the samples for the last run
                if id_ann == annealing_set.shape[0] - 1:
                    if (i + 1) % skip == 0:
                        res.append(theta1.clone())
        return res

    # Free GPU memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()


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

    prop_mean = theta_pre.mean(dim = 0)
    prop_cov = torch.diag(inf_rate) @ torch.cov(theta_pre.T) @ torch.diag(inf_rate)

    sample_size = 1000
    theta_r0 = sample_trunc_fast(prop_mean, prop_cov, lower, upper, sample_size, batch_size=10000)

    theta_init = theta_r0[:1000]


    epis = 0.01 / data_obs.shape[0] 
    S = 10000 
    skip = 1000
    annealing_set = torch.cat( (torch.linspace(0.1, 1.0, 10), torch.ones(2)) )  
    res = repar_draw_post_vec_annealing3(skip, annealing_set, lower.to(device), upper.to(device), data_obs, theta_init, epis, S)
    theta_r1 = torch.cat(res, dim=0).to("cpu")

    A = get_A(M)
    grids = torch.arange(0, 1.01, 0.01)
    psi_grids = get_psi(grids, M)

    # the predicted y across all the preconditioning solutions
    pred_ys_ann_repar = psi_grids @ torch.linalg.inv(A) @ theta_r1.T


    # make sure the folder exists and then save
    os.makedirs(f'sample_res_nmodel_fisher_nondiag', exist_ok=True)

    pd.DataFrame( pred_ys_ann_repar.numpy() ).to_csv(f'sample_res_nmodel_fisher_nondiag/pred_ys_ann_task{task_id}.csv', index=False, header=False)
    np.save(f'sample_res_nmodel_fisher_nondiag/theta_init_task{task_id}.npy', theta_init.cpu().numpy())
    np.save(f'sample_res_nmodel_fisher_nondiag/theta_r1_task{task_id}.npy', theta_r1.cpu().numpy())

    t2 = time.time()
    print(f"Annealing used {(t2-t1)/60} mins")



if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
