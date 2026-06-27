from utils_SDEMEM import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path

import ot

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # seems faster

# ===== Setting ===== #
obs_size = 200
T = 30
theta_dim = 12
x_dim = 180

prior_mean = torch.tensor([5, 1, 3, -1.5, -0.694, -3, 0.027, 0, -0.8, -0.8, -0.8, -0.8], dtype=torch.float32)
prior_std = torch.tensor([1, 1, 1, 1, 0.6, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
theta_true = torch.tensor([5.7, 0.7, 2.08, -1.6, -0.694, -3, 0.027, 0, -1.15, -1.15, -1.15, -1.15], dtype=torch.float32).reshape(1, -1)



# =============== Load configurations =============== #
task_id = int(sys.argv[1])

num_theta = 105000  # 10 batches (sets) of theta candidates, and the total number of theta is 1.05e6
# batch_id = int(config["batch_id"]) range(10)
theta_batch_size = 1000 # generate data in batches to speed up


def main(task_id, batch_id):
    print(f"Task {task_id}, batch {batch_id}, num_theta = {num_theta}, theta_batch_size = {theta_batch_size}")

    t1 = time.time()


    # ABC with W1
    def ABC_W1(data_obs, theta_set, theta_batch_size):
        """
        ABC by comparing W1(data_obs, data_simu)
        """
        data_obs = data_obs.to(device)
        theta_set = theta_set.to(device)
        W1_set = torch.zeros(theta_set.shape[0])
        marg1 = (1 / data_obs.shape[0]) * torch.ones(data_obs.shape[0]).to(device)

        for batch_start in range(0, theta_set.shape[0], theta_batch_size):
            time_start = time.time()
            batch_end = min(batch_start + theta_batch_size, theta_set.shape[0])
            

            theta_batch = theta_set[batch_start:batch_end]
            theta_batch_rep = theta_batch.repeat_interleave(obs_size, dim=0)
            data_simu_batch = gen_x_given_theta(theta_batch_rep, T=T, mute=True).reshape(theta_batch.shape[0], obs_size, -1)

            for i in range(theta_batch.shape[0]):
                data_simu = data_simu_batch[i]
                marg2 = (1 / data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device)
                cost_mat = ot.dist(data_obs, data_simu, metric='euclidean')
                W1_set[batch_start + i] = ot.emd2(marg1, marg2, cost_mat)

            time_end = time.time()
            print(f"Finished theta batches: {batch_start + 1}-{batch_end}/{theta_set.shape[0]} (Time: {time_end - time_start:.2f}s)")

        return W1_set


    # Load SW data to generate theta candidates
    theta_SW1 = np.load(f"res_SW1/theta_SW1_task{task_id % 10}.npy")[:100]
    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)

    prop_mean = theta_SW1.mean(dim=0, keepdims=True)
    prop_std = theta_SW1.std(dim=0, keepdims=True)

    # inflate the proposal std
    prop_std *= 2

    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")


    # ========== load observed data
    x_obs = torch.from_numpy(np.load(f"data_obs/x_obs_task{task_id}.npy"))
    print(x_obs.shape)


    # ============ Run ABC W1
    theta_set = prop_mean + prop_std * torch.randn(num_theta, theta_dim)
    W1_set = ABC_W1(x_obs, theta_set, theta_batch_size)


    # Save both theta_set and W1_set
    save_dir = Path("res_ABC")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"theta_set{task_id}_batch{batch_id}.npy", theta_set.detach().cpu().numpy())
    np.save(save_dir / f"W1_set{task_id}_batch{batch_id}.npy", W1_set.detach().cpu().numpy())


    t2 = time.time()
    print(f"ABCW1 used {(t2 - t1) / 60} mins")


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    for batch_id in range(10):  # 10 batches, can be modified to run in parallel
        main(task_id, batch_id)