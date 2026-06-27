from utils_SDEMEM_realdata import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path

import ot

device = torch.device("cpu") # seems faster

# ===== Setting for real data ===== #
T = 30
theta_dim = 12
x_dim = 180 

obs_size = 40

# =============== Load configurations =============== #
num_theta = int(1.25e5) # int(config["num_theta"]) # 10 batches, and the total num is 1.25e6


def main(task_id):
    t1 = time.time()
    #   ABC with W1   
    def ABC_W1(data_obs, theta_set):
        """
        ABC by comparing W1(data_obs, data_simu)
        """
        data_obs = data_obs.to(device)
        theta_set = theta_set.to(device)
        W1_set = torch.zeros(theta_set.shape[0])
        
        for i in range(theta_set.shape[0]):
            if i % 100 == 0:
                print(f"Finished: {i+1}/{theta_set.shape[0]}")

            # generate simulated data based on theta_i
            theta = theta_set[i].to(device)

            data_simu = gen_x_given_theta(theta.repeat(obs_size, 1), T=T, mute = True)
            
            # calculate W1(data_obs, data_simu)
            marg1 = (1/data_obs.shape[0]) * torch.ones(data_obs.shape[0]).to(device) # marginal distribution
            marg2 = (1/data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device)
            cost_mat = ot.dist(data_obs.to(device), data_simu, metric='euclidean')  
            W1_set[i] = ot.emd2(marg1, marg2, cost_mat)
            # W1_set[i] = ot.sinkhorn2(marg1, marg2, cost_mat, 0.1) # entropy regularized, not faster in this case (n = 1000)
        return W1_set


    # Load SW data
    theta_SW1 = np.load("res_SW1/theta_SW1.npy")
    loss_SW1 = np.load("res_SW1/final_loss.npy")

    nan_idx = np.isnan(theta_SW1).any(axis=1)
    theta_SW1 = theta_SW1[~nan_idx]
    loss_SW1 = loss_SW1[~nan_idx]
    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)[:100]
    print(f"theta_SW1.shape = {theta_SW1.shape}")



    prop_mean = theta_SW1.mean(dim = 0, keepdims = True)
    prop_std = theta_SW1.std(dim = 0, keepdims = True) 

    # inflate the proposal std
    prop_std *= 2

    # the previous prop_std is too small for these two dimensions
    prop_std[0, 0] *= 3
    prop_std[0, 6] *= 3


    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")


    # ========== load observed data
    df = pd.read_excel("realdata/20160427_mean_eGFP.xlsx", header=None)
    x_obs = torch.tensor(df.to_numpy(), dtype=torch.float32)[:, 1:].T.log()
    x_obs = x_obs[:obs_size]

    print(f"x_obs.shape = {x_obs.shape}")


    # ============ Run ABC W1
    theta_set = prop_mean + prop_std * torch.randn(num_theta, theta_dim)
    W1_set = ABC_W1(x_obs, theta_set)


    # Save both theta_set and W1_set
    save_dir = Path(f"res_ABC")
    save_dir.mkdir(parents=True, exist_ok=True)  
    np.save(save_dir / f"theta_set_batch{task_id}.npy", theta_set.detach().cpu().numpy())
    np.save(save_dir / f"W1_set_batch{task_id}.npy", W1_set.detach().cpu().numpy())


    t2 = time.time()
    print(f"ABCW1 used {(t2-t1)/60} mins")


if __name__ == "__main__":
    for task_id in range(10):
        main(task_id)

