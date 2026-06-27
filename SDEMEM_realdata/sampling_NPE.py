import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
from pathlib import Path
import pandas as pd
import torch
from utils_SDEMEM_realdata import *
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding


# ===== Setting for real data ===== #
var_name = ['log_m0', 'log_scale', 'log_offset', 'log_sigma', 'mu_delta', 'mu_gamma', 'mu_k', 'mu_t0', 'log_tau_delta', 'log_tau_gamma', 'log_tau_k', 'log_tau_t0']


T = 30
theta_dim = 12
x_dim = 180 
obs_size = 40


task_id = 0


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
        output_dim = int(config["summary_dim"]),
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



    weights_path = Path("NPE_embed_model_newdeepsets") / f"npe_net_weights_task{task_id}.pth"
    state_dict = torch.load(weights_path, map_location=device)
    inference._neural_net.load_state_dict(state_dict)
    inference._neural_net.eval()

    posterior = inference.build_posterior(inference._neural_net)


    # ========== load observed data
    df = pd.read_excel("realdata/20160427_mean_eGFP.xlsx", header=None)
    x_obs = torch.tensor(df.to_numpy(), dtype=torch.float32)[:, 1:].T.log() 
    x_obs = x_obs[:obs_size]


    num_samples = 8000
    samples = posterior.sample((num_samples,), x=x_obs.unsqueeze(0), show_progress_bars=True)

    os.makedirs("sample_res_all", exist_ok=True)
    np.save(f"sample_res_all/theta_post_NPE_newdeepsets_task{task_id}.npy", samples.cpu().numpy())


if __name__ == "__main__":
    main()
