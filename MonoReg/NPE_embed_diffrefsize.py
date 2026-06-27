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

start_time = time.time()
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



# =============== Load configurations =============== #
# config = json.loads(sys.argv[2])
task_id = 1 # int(config["task_id"])
num_epochs = 3000 # int(config["num_epochs"])
batch_size = 200 # int(config["batch_size"]) 
summary_dim = 128 # int(config["summary_dim"]) 
embed_hidden_size = 128 # int(config["embed_hidden_size"])
embed_num_layers = 3 # int(config["embed_num_layers"]) # 3


def main(sample_size):
    # ===== generate training data ===== #
    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()


    ### generate training data
    lower = torch.zeros(M + 1) # .to(device)
    lower[0] = a0
    lower[1:] = a
    upper = torch.zeros(M + 1) # .to(device)
    upper[0] = b0
    upper[1:] = b



    theta_r0, data_r0 = gen_ref_prior(lower, upper, obs_size, sample_size + 100) # gen_ref(mean_theta, std_theta, lower, upper, obs_size, sample_size + 100)


    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask][:sample_size]
    data_r0 = data_r0[~bad_mask][:sample_size].reshape(sample_size, obs_size, -1)

    print(f'Ref size (train+val): theta_r0.shape = {theta_r0.shape}, data_r0.shape = {data_r0.shape}')



    # ====== Construct the embedding net ====== #
    single_trial_net = FCEmbedding(
        input_dim = x_dim,
        num_hiddens = embed_hidden_size,
        num_layers = embed_num_layers,
        output_dim = summary_dim,
    )

    embedding_net = PermutationInvariantEmbedding(
        single_trial_net,
        trial_net_output_dim=summary_dim,
        num_hiddens = embed_hidden_size,
        num_layers = embed_num_layers,
        output_dim = summary_dim,
    )

    # Use a normalizing flow as the density estimator
    density_estimator = posterior_nn("maf", embedding_net=embedding_net)


    # ====== Train the model ====== #
    inference = NPE(density_estimator=density_estimator, device = device)
    # NOTE: we don't exclude invalid x because we used NaNs for the missing trials.
    inference.append_simulations(
        theta_r0.to(device),
        data_r0.to(device),
        # exclude_invalid_x=False,
    ).train(
        training_batch_size=batch_size,
        max_num_epochs=num_epochs,
        validation_fraction=0.5, 
        show_train_summary=True
        )
    # posterior = inference.build_posterior()

    print(inference._summary)


    # Save the NPE model
    save_dir_nn = Path(f"NPE_embed_model_diffrefsize")
    save_dir_nn.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save(inference._neural_net.state_dict(), save_dir_nn / f"npe_net_weights_task{task_id}_ref{sample_size}.pth")



    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')

if __name__ == "__main__":
    for sample_size in [250000, 100000, 50000, 25000]:
        main(sample_size)