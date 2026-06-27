from utils_SDEMEM import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
from pathlib import Path


start_time = time.time()
# ===== Setting ===== #
obs_size = 200
T = 30
theta_dim = 12
x_dim = 180

prior_mean = torch.tensor([5, 1, 3, -1.5, -0.694, -3, 0.027, 0, -0.8, -0.8, -0.8, -0.8], dtype = torch.float32)
prior_std = torch.tensor([1, 1, 1, 1, 0.6, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype = torch.float32)

# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
theta_true = torch.tensor([5.7, 0.7, 2.08, -1.6, -0.694, -3, 0.027, 0, -1.15, -1.15, -1.15, -1.15], dtype = torch.float32).reshape(1, -1)



# =============== Load configurations =============== #
sample_size = int(1.05e6) # int(config["sample_size"]) 


def main(task_id):
    # ===== generate training data ===== #
    #################
    # Training Data #
    #################
    # Load SW data
    theta_SW1 = np.load(f"res_SW1/theta_SW1_task{task_id}.npy")[:100]
    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)

    prop_mean = theta_SW1.mean(dim = 0, keepdims = True)
    prop_std = theta_SW1.std(dim = 0, keepdims = True) 

    # inflate the proposal std
    prop_std *= 2


    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")

    # ===== ref_R ===== #
    # 'extra' just means each simulation is a dataset of size n=obs_size
    path_theta = Path(f'ref_NPE/theta_r0_extra_task{task_id}.npy')
    path_x = Path(f'ref_NPE/x_r0_extra_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists

    if not (path_theta.exists() and path_x.exists()):
        theta_r0_extra = torch.empty(sample_size, theta_dim)
        x_r0_extra = torch.empty(sample_size, obs_size * x_dim)

        step = max(1, int(sample_size / 3))
        kept = 0

        for start in range(0, sample_size, step):
            end = min(start + step, sample_size)
            current_n = end - start

            theta_part = prop_mean + prop_std * torch.randn(current_n, theta_dim)
            x_part = torch.empty(current_n, obs_size * x_dim)

            for i in range(obs_size):
                x_part[:, i * x_dim:(i + 1) * x_dim] = gen_x_given_theta(theta_part.to(device), T=T).cpu()

            valid_mask = torch.isfinite(x_part).all(dim=1)
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                continue

            theta_r0_extra[kept:kept + num_valid].copy_(theta_part[valid_mask])
            x_r0_extra[kept:kept + num_valid].copy_(x_part[valid_mask])
            kept += num_valid

        theta_r0_extra = theta_r0_extra[:kept]
        x_r0_extra = x_r0_extra[:kept]

        print(f"generated reference table with shape theta: {theta_r0_extra.shape}, x: {x_r0_extra.shape}")

        np.save(path_theta, theta_r0_extra.numpy())
        np.save(path_x, x_r0_extra.numpy())


    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)