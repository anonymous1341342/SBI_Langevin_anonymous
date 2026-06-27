import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
from utils_queuing_single import *
import matplotlib.pyplot as plt
import torch
from torch import eye, zeros

from sbi.analysis import pairplot
from sbi.inference import NLE
from sbi.utils import BoxUniform

from pathlib import Path
import sys
import json



start_time = time.time()
# ================ Setting =============== #
a1 = 0.0
b1 = 10.0

a2 = 0.0
b2 = 10.0

a3 = 0.01 # 0.0
b3 = 0.5

theta_dim = 3
x_dim = 5

# =============== Load configurations =============== #
# config = json.loads(sys.argv[2])
num_epochs = 30 # config["num_epochs"] 
batch_size = 200 # int(config["batch_size"]) 


def main(task_id):
    # ===== generate training data ===== #
    sample_size = int(20000 * 500)
    theta_r0, x_r0 = gen_ref_table_distinct_theta(a1, a2, a3, b1, b2, b3, dim = 5, sample_size = sample_size)
    print(f'Shape of theta and x: {theta_r0.shape}, {x_r0.shape}')


    # ====== Train the model ====== #
    prior = BoxUniform(low=torch.tensor([a1, a2, a3], dtype=torch.float32, device = device), high=torch.tensor([b1, b2, b3], dtype=torch.float32, device = device))

    inference = NLE(prior, show_progress_bars=True, density_estimator="maf", device = device)
    inference.append_simulations(
        theta_r0.to(device), 
        x_r0.to(device)).train(
            training_batch_size=batch_size,
            validation_fraction=0.1, 
            show_train_summary=True,
            max_num_epochs=num_epochs,
            stop_after_epochs=10
            )

    print(inference._summary)


    # Save the NLE model
    save_dir_nn = Path(f"NLE_model_001")
    save_dir_nn.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save(inference._neural_net.state_dict(), save_dir_nn / f"nle_net_weights_task{task_id}.pth")

    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)