import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils_SI_5F import *
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
        prior_score = -3.0 - log_theta0
        
        log_theta1 = log_theta0 + epis * (like_score_hat + prior_score) + np.sqrt(2.0 * epis) * torch.randn(log_theta0.shape).to(device) # draw a new sample
        log_theta0 = log_theta1 
    return log_theta1


def main():

    input_size = (K+2) + (K+2)*T
    output_size = K+2

    for model_id in tqdm(range(10)):
        model = ELU_Nonadd_Medium(input_size, output_size).to(device)
        checkpoint = torch.load(f'nmodel/checkpoint_task{model_id}.pth', map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        bias_lastlayer = checkpoint['bias_lastlayer']
        with torch.no_grad(): 
            model.layers[-1].bias -= bias_lastlayer
            
        for idx in range(10): # one model takes 10 observed data
            obs_id = 10 * model_id + idx
            data_obs = torch.tensor(np.load(f"data_obs/data_obs_task{obs_id}.npy"), dtype=torch.float32)
            SS_obs = get_SS(data_obs)
            
            log_theta_init = -3.0 + torch.randn(10000, K+2)
            epis = 0.1 / data_obs.shape[0] # 0.001 / 364
            S = 5000 
            
            log_theta_r1 = draw_post(model, SS_obs, log_theta_init, epis, S)
            post_samples = log_theta_r1.exp().cpu()
            # ensure the directory exists
            os.makedirs("Langevin_res", exist_ok=True)
            np.save(f"Langevin_res/post_samples_obs{obs_id}.npy", post_samples.cpu().numpy())

if __name__ == "__main__":
    main()