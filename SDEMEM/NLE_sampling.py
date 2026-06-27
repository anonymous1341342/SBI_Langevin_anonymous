from utils_SDEMEM import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path

from sbi.analysis import pairplot
from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters

from pathlib import Path
import sys
import json



def main(obs_id):
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
    task_id = obs_id % 10 # 1 model takes 10 observed data

    theta_dummy = torch.rand(100, theta_dim).to(device)
    x_dummy = torch.rand(100, x_dim).to(device)


    from torch.distributions import MultivariateNormal

    # ====== Load the trained NLE model ====== #
    prior = MultivariateNormal(loc=prior_mean.to(device), covariance_matrix=torch.diag(prior_std**2).to(device))

    inference = NLE(prior, show_progress_bars=True, density_estimator="maf", device = device)


    # Build the neural net once so the modules exist before loading state_dict.
    # We use a tiny dummy batch only for initializing the neural networks
    inference.append_simulations(
        theta_dummy.to(device), 
        x_dummy.to(device)).train(
            training_batch_size=10,
            max_num_epochs=1
            )


    weights_path = Path("NLE_model") / f"nle_net_weights_task{task_id}.pth"
    state_dict = torch.load(weights_path, map_location=device)
    inference._neural_net.load_state_dict(state_dict)
    inference._neural_net.eval()


    mcmc_parameters = MCMCPosteriorParameters(
        method='nuts_pyro', 
        num_chains=4,
        # num_workers=4,
        thin=1,
        warmup_steps=500,
        init_strategy="proposal"
    )


    posterior = inference.build_posterior(
        posterior_parameters=mcmc_parameters
    )



    x_obs = torch.from_numpy(np.load(f"data_obs/x_obs_task{obs_id}.npy")).to(device)
    theta_post = posterior.sample(sample_shape=(2000,), x=x_obs.to(device))


    save_dir = Path(f"res_NLE")
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    np.save(save_dir / f"theta_post_task{obs_id}.npy", theta_post.cpu().numpy())


    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')


if __name__ == "__main__":
    obs_id = int(sys.argv[1])
    main(obs_id)