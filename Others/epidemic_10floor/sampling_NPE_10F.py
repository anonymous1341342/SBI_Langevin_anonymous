import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils_npe import *
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
    for task_id in range(50):
        data_obs = pd.read_csv(f"data_obs/y_obs_task{task_id}.csv")
        data_obs = torch.tensor(data_obs.values, dtype=torch.float32).contiguous().to(device)
        SS_obs = get_SS(data_obs) 
        model_NPE = torch.load(f'logNPE_model/model_task{task_id}.pth').to(device)
        # NPE
        model_NPE.eval()
        with torch.no_grad():
            mu, sigma = model_NPE(SS_obs.view(1, -1))
        mu = mu.ravel().cpu()
        Cov = (sigma @ sigma.T).cpu()

        os.makedirs("NPE_res", exist_ok=True)
        np.save(f"NPE_res/mu_obs{task_id}.npy", mu.cpu().numpy())
        np.save(f"NPE_res/Cov_obs{task_id}.npy", Cov.cpu().numpy())

if __name__ == "__main__":
    main()
