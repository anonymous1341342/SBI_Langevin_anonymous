import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
import sys
import json
import pandas as pd
import torch
from utils_queuing_single import *
from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters

device = torch.device("cpu") # it seems we can only use cpu for sampling in the sbi package


def main(obs_id):
    # ================ Setting =============== #
    a1 = 0.0
    b1 = 10.0

    a2 = 0.0
    b2 = 10.0

    a3 = 0.01 # 0.0
    b3 = 0.5

    theta_dim = 3
    x_dim = 5
    obs_size = 500


    # ====== Load the trained NLE model ====== #
    prior = BoxUniform(low=torch.tensor([a1, a2, a3], dtype=torch.float32, device = device), high=torch.tensor([b1, b2, b3], dtype=torch.float32, device = device))
    # prior.to(device) #inplace
    inference = NLE(prior, show_progress_bars=True, density_estimator="maf", device = device)


    # Build the neural net once so the modules exist before loading state_dict.
    # We use a tiny dummy batch only for initializing the neural networks
    theta_dummy, x_dummy = gen_ref_table_distinct_theta(a1, a2, a3, b1, b2, b3, dim = 5, sample_size = 100)

    inference.append_simulations(
        theta_dummy.to(device), 
        x_dummy.to(device)).train(
            training_batch_size=10,
            max_num_epochs=1
            )

    task_id = obs_id // 10 # one model takes 10 observed data

    weights_path = Path("NLE_model_001") / f"nle_net_weights_task{task_id}.pth"
    state_dict = torch.load(weights_path, map_location=device)
    inference._neural_net.load_state_dict(state_dict)
    inference._neural_net.eval()

    mcmc_parameters = MCMCPosteriorParameters(
        method='nuts_pyro', 
        num_chains=4,
        num_workers=4,
        thin=1,
        warmup_steps=500,
        init_strategy="proposal"
    )

    posterior = inference.build_posterior(
        posterior_parameters=mcmc_parameters
    )



    save_dir = Path(f"res_inference/NLE")
    save_dir.mkdir(parents=True, exist_ok=True)  

    if not (save_dir / f"samples{obs_id}.npy").exists():
        x_obs = pd.read_csv(f"data_obs/x_obs_task{obs_id}.csv")
        x_obs = torch.tensor(x_obs.values, dtype=torch.float32).contiguous()
        theta_post = posterior.sample(sample_shape=(2000,), x=x_obs.to(device))

        # save the posterior samples
        np.save(save_dir / f"samples{obs_id}.npy", theta_post.detach().cpu().numpy())


if __name__ == "__main__":
    obs_id = int(sys.argv[1])
    main(obs_id)