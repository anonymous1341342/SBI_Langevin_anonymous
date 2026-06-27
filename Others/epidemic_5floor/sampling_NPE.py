import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils_npe import *
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


def main():
    for model_id in range(10):
        model_NPE = torch.load(f'NPE_model/model_task{model_id}.pth').to(device)
        for idx in range(10): # one model takes 10 observed data
            obs_id = 10 * model_id + idx
        
            data_obs = torch.tensor(np.load(f"data_obs/data_obs_task{obs_id}.npy"), dtype=torch.float32)
            SS_obs = get_SS(data_obs)
            
            # results for NPE
            model_NPE.eval()
            with torch.no_grad():
                mu, sigma = model_NPE(SS_obs.view(1, -1))
            mu = mu.ravel().cpu()
            Cov = (sigma @ sigma.T).cpu()

            # ensure the output directory exists
            os.makedirs("NPE_res", exist_ok=True)

            np.save(f"NPE_res/mu_obs{obs_id}.npy", mu.cpu().numpy())
            np.save(f"NPE_res/Cov_obs{obs_id}.npy", Cov.cpu().numpy())


if __name__ == "__main__":
    main()

    