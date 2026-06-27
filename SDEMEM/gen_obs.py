from utils_SDEMEM import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ===== Setting ===== #
obs_size = 200
T = 30
theta_dim = 12
x_dim = 180

prior_mean = torch.tensor([5, 1, 3, -1.5, -0.694, -3, 0.027, 0, -0.8, -0.8, -0.8, -0.8], dtype = torch.float32)
prior_std = torch.tensor([1, 1, 1, 1, 0.6, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype = torch.float32)

# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
theta_true = torch.tensor([5.7, 0.7, 2.08, -1.6, -0.694, -3, 0.027, 0, -1.15, -1.15, -1.15, -1.15], dtype = torch.float32).reshape(1, -1)

def main(task_id):
    save_dir = Path("data_obs")
    save_dir.mkdir(parents=True, exist_ok=True)


    x_obs = gen_x_given_theta(theta_true.repeat(obs_size, 1), T = T, epis = 0.01, len_interpolate = 1/6, seed = task_id + 1000) 
    assert torch.isfinite(x_obs).all(), f"Non-finite values found in x_obs at task_id={task_id}"
    np.save(save_dir / f"x_obs_task{task_id}.npy", x_obs.detach().cpu().numpy())
    

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)

