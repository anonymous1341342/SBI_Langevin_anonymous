import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
from pathlib import Path
import pandas as pd
import torch
from utils_SDEMEM import *
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding


# ===== Setting ===== #
obs_size = 200
T = 30
theta_dim = 12
x_dim = 180

prior_mean = torch.tensor([5, 1, 3, -1.5, -0.694, -3, 0.027, 0, -0.8, -0.8, -0.8, -0.8], dtype = torch.float32)
prior_std = torch.tensor([1, 1, 1, 1, 0.6, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype = torch.float32)

# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
theta_true = torch.tensor([5.7, 0.7, 2.08, -1.6, -0.694, -3, 0.027, 0, -1.15, -1.15, -1.15, -1.15], dtype = torch.float32).reshape(1, -1)

var_name = ['log_m0', 'log_scale', 'log_offset', 'log_sigma', 'mu_delta', 'mu_gamma', 'mu_k', 'mu_t0', 'log_tau_delta', 'log_tau_gamma', 'log_tau_k', 'log_tau_t0']


def main():
    theta_dummy = torch.rand(100, theta_dim)
    x_dummy = torch.rand(100, obs_size, x_dim)


    config = {
        "batch_size": 200,
        "summary_dim": 128,
        "embed_hidden_size": 128,
        "embed_num_layers": 3,
    }

    single_trial_net = FCEmbedding(
        input_dim=x_dim,
        num_hiddens=int(config["embed_hidden_size"]),
        num_layers=int(config["embed_num_layers"]),
        output_dim=int(config["summary_dim"]),
    )

    embedding_net = PermutationInvariantEmbedding(
        single_trial_net,
        trial_net_output_dim=int(config["summary_dim"]),
        num_hiddens = int(config["embed_hidden_size"]),
        num_layers = int(config["embed_num_layers"]),
        output_dim = int(config["summary_dim"])
    )

    density_estimator = posterior_nn("maf", embedding_net=embedding_net)
    inference = NPE(density_estimator=density_estimator)

    # Build the neural net once so the modules exist before loading state_dict.
    # We use a tiny dummy batch only for initializing the neural networks
    inference.append_simulations(theta_dummy, x_dummy).train(
        max_num_epochs=1,
        training_batch_size=2,
        validation_fraction=0.5,
        show_train_summary=False,
    )




    # ensure the folder 'res_NPE' exists
    Path("res_NPE_newdeepsets").mkdir(exist_ok=True)
    for obs_id in range(100):
        task_id = obs_id % 10
            
        print(f'\ntask_id: {task_id}')
        weights_path = Path("NPE_embed_model_newdeepsets") / f"npe_net_weights_task{task_id}.pth"
        state_dict = torch.load(weights_path, map_location=device)
        inference._neural_net.load_state_dict(state_dict)
        inference._neural_net.eval()

        posterior = inference.build_posterior(inference._neural_net)

        x_obs = torch.from_numpy(np.load(f"data_obs/x_obs_task{obs_id}.npy"))
        x_obs = x_obs.reshape(1, obs_size, x_dim)

        num_samples = 10000
        samples = posterior.sample((num_samples,), x=x_obs, show_progress_bars=True)
        np.save(f"res_NPE_newdeepsets/samples_task{obs_id}.npy", samples.cpu().numpy())


if __name__ == "__main__":
    main()