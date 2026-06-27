import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils_SI10F_scomat import *
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    def draw_post(model, x_obs, log_theta_init, epis = 0.001, S = 100):
        # vectorized, generate multiple MC chains, but only return the last draws of each chain
        # epis: step size
        # S: length of each chain
        # log_theta_init: dim m*d
        model.to(device)
        model.eval()
        
        log_theta0 = log_theta_init.to(device) # initial value of log_theta
        x_obs = x_obs.to(device).view(1, -1)
        for i in range(S):
            like_score_hat = model(log_theta0, x_obs.repeat(log_theta0.shape[0], 1)).detach().to(device)
            prior_score = ( -3.0 - log_theta0 ) / 4
            
            log_theta1 = log_theta0 + epis * (like_score_hat + prior_score) + np.sqrt(2.0 * epis) * torch.randn(log_theta0.shape).to(device) # draw a new sample
            log_theta0 = log_theta1 
        return log_theta1



    input_size = (K+2) + (K+2)*T
    output_size = K+2

    for task_id in range(50):
        model = ELU_Nonadd_Medium(input_size, output_size).to(device)
        checkpoint = torch.load(f'model_10F/checkpoint_task{task_id}.pth', map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        bias_lastlayer = checkpoint['bias_lastlayer']
        with torch.no_grad(): 
            model.layers[-1].bias -= bias_lastlayer
        
        data_obs = pd.read_csv(f"data_obs/y_obs_task{task_id}.csv")
        data_obs = torch.tensor(data_obs.values, dtype=torch.float32).contiguous().to(device)
        SS_obs = get_SS(data_obs) 

        precond_samples = pd.read_csv(f"res_precond/pre_samples_lam0_task{task_id}.csv")
        precond_samples = torch.tensor(precond_samples.values, dtype=torch.float32).contiguous()
        sample_size = 10000 
        mean_theta = precond_samples.mean(dim = 0)
        std_theta = precond_samples.std(dim = 0)
        mu_new = mean_theta.view(1, -1).repeat(sample_size, 1)
        sigma_new = std_theta.view(1, -1).repeat(sample_size, 1)
        log_theta_r0 = mu_new + sigma_new * torch.randn(mu_new.shape)

        # Langevin sampling
        epis = 0.1 / data_obs.shape[0] 
        S = 10000 
        log_theta_init = log_theta_r0
        log_theta_r1 = draw_post(model, SS_obs, log_theta_init, epis, S)
        post_samples = log_theta_r1.exp().cpu()

        # ensure the directory exists
        os.makedirs("Langevin_res", exist_ok=True)
        np.save(f"Langevin_res/post_samples_obs{task_id}.npy", post_samples.cpu().numpy())


if __name__ == "__main__":
    main()
        