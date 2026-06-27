from utils_queuing_single import *
import pandas as pd
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    for task_id in range(100):
        # Create the set of x_obs and fix them for inference
        path_x_obs = Path(f"data_obs/x_obs_task{task_id}.npy")
        path_x_obs.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists
        if not (path_x_obs.exists()):
            theta_true = [1.0, 4.0, 0.2]
            x_obs = gen_obs_data(*theta_true, dim = 5, obs_size = 500)
            df_x_obs = pd.DataFrame(x_obs.cpu())
            df_x_obs.to_csv(f"data_obs/x_obs_task{task_id}.csv", index=False)

if __name__ == "__main__":
    main()