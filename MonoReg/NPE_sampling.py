from utils_nmodel import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.inference import NPE
from pathlib import Path
import sys
import json
import time
import pandas as pd



# ================ Setting =============== #
# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0

M = 10
x_dim = 2


sigma = 0.1 
obs_size = 1000



def main():
    theta_dummy = torch.rand(100, M + 1)
    x_dummy = torch.rand(100, obs_size, 2)

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
        num_hiddens=int(config["embed_hidden_size"]),
        num_layers=int(config["embed_num_layers"]),
        output_dim=int(config["summary_dim"]),
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


    for task_id in range(10):
        weights_path = Path("NPE_embed_model_newdeepsets") / f"npe_net_weights_task{task_id}.pth"
        state_dict = torch.load(weights_path, map_location=device)
        inference._neural_net.load_state_dict(state_dict)
        inference._neural_net.eval()

        posterior = inference.build_posterior(inference._neural_net)

        data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
        data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous().reshape(1, obs_size, x_dim)

        num_samples = 10000
        theta_post = posterior.sample((num_samples,), x=data_obs, show_progress_bars=True)

        # save theta_post
        save_dir = Path(f"res_NPE_newdeepsets") 
        save_dir.mkdir(parents=True, exist_ok=True)  
        np.save(save_dir / f"theta_post{task_id}.npy", theta_post.detach().cpu().numpy())


if __name__ == "__main__":
    main()