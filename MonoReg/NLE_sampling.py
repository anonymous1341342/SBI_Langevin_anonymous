import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import torch
from utils_monoBP_single import *

from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
from pathlib import Path
import sys
import json




def main(task_id):
    start_time = time.time()

    sigma = 0.1 # noise level
    obs_size = 1000
    # prior for theta0
    a0 = -5.0
    b0 = 5.0
    # prior for theta1-thetaM
    a = 0.0
    b = 1.0



    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous()

    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()

    x_obs = data_obs[:, 0]
    y_obs = data_obs[:, 1]


    lower = torch.zeros(M + 1) # .to(device)
    lower[0] = a0
    lower[1:] = a
    upper = torch.zeros(M + 1) # .to(device)
    upper[0] = b0
    upper[1:] = b

    actual_inf_rate = torch.ones(M + 1)
    actual_inf_rate[-2:] = 2

    inf_rate = torch.zeros(M + 1)
    for i in range(M + 1):
        inf_rate[i] = get_inf_rate(mode = theta_pre.mean(dim = 0)[i].item(), std_orig = theta_pre.std(dim = 0)[i].item(),
                        lower = lower[i].item(), upper = upper[i].item(), actual_inf_rate = actual_inf_rate[i].item())


    mean_theta = theta_pre.mean(dim = 0)
    std_theta = inf_rate * theta_pre.std(dim = 0)


    theta_dummy = torch.rand(100, 11).to(device)
    x_dummy = torch.rand(100, 2).to(device)


    # ====== Load the trained NLE model ====== #
    prior = prior = BoxUniform(low = lower.to(device), high = upper.to(device))
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
        num_chains= 4,
        num_workers=1,
        thin=1,
        warmup_steps=500,
        init_strategy="proposal"
    )

    posterior = inference.build_posterior(
        posterior_parameters=mcmc_parameters
    )



    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous()


    theta_post = posterior.sample(sample_shape=(1000,), x=data_obs.to(device))


    # Save theta_post
    save_dir = Path(f"res_NLE")
    save_dir.mkdir(parents=True, exist_ok=True)  
    np.save(save_dir / f"theta_post{task_id}.npy", theta_post.detach().cpu().numpy())



    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)