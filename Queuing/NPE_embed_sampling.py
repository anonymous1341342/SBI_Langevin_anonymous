import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
from pathlib import Path

import pandas as pd
import torch
from utils_queuing_nmodel import *
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding



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

def main():
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
    )

    density_estimator = posterior_nn("maf", embedding_net=embedding_net)
    inference = NPE(density_estimator=density_estimator)

    # Build the neural net once so the modules exist before loading state_dict.
    # We use a tiny dummy batch only for initializing the neural networks
    theta_dummy, x_dummy = gen_ref_table(a1, a2, a3, b1, b2, b3, dim = 5, obs_size = obs_size, sample_size = 100)
    x_dummy = x_dummy.reshape(100, obs_size, -1)

    inference.append_simulations(theta_dummy, x_dummy).train(
        max_num_epochs=1,
        training_batch_size=2,
        validation_fraction=0.5,
        show_train_summary=False,
    )



    # ensure the folder "NPE_res" exists
    Path("NPE_res").mkdir(exist_ok=True)

    for obs_id in range(100):
        model_id = obs_id // 10
        weights_path = Path("NPE_embed_model") / f"npe_net_weights_task{model_id}.pth"
        state_dict = torch.load(weights_path, map_location=device)
        inference._neural_net.load_state_dict(state_dict)
        inference._neural_net.eval()

        posterior = inference.build_posterior(inference._neural_net)

        x_obs = pd.read_csv(f"data_obs/x_obs_task{obs_id}.csv")
        x_obs = torch.tensor(x_obs.values, dtype=torch.float32).contiguous()

        # The embedding net expects shape (batch, num_trials, x_dim).
        x_obs = x_obs.reshape(1, obs_size, x_dim)

        num_samples = 10000
        samples = posterior.sample((num_samples,), x=x_obs.cpu(), show_progress_bars=False)
        # samples[:, 1] = samples[:, 0] + samples[:, 1]

        np.save(f"NPE_res/samples_obs{obs_id}.npy", samples.cpu().numpy())


if __name__ == "__main__":
    main()