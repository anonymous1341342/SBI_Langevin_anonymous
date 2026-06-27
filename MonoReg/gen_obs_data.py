import torch
import pandas as pd
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sigma = 0.1 
obs_size = 1000


def main(task_id):
    x_obs = torch.rand(obs_size)
    y_obs = torch.tanh(4.0 * x_obs - 2.0) + sigma * torch.randn(obs_size)

    data_obs = torch.cat((x_obs.view(-1, 1), y_obs.view(-1, 1)), dim=1)

    # ensure the directory exists
    os.makedirs("data_obs", exist_ok=True)

    # save the data as csv
    pd.DataFrame(data_obs.cpu().numpy()).to_csv(
        f"data_obs/data_obs_task{task_id}.csv",
        index=False
    )


if __name__ == "__main__":
    task_id = sys.argv[1]  
    main(task_id)