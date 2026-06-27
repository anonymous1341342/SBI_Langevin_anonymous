import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
from utils_queuing_nmodel import *
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.inference import NPE
from pathlib import Path
import sys
import json

start_time = time.time()
# ================ Setting =============== #
a1 = 0.0
b1 = 10.0

a2 = 0.0
b2 = 10.0

a3 = 0.01 # 0.0
b3 = 0.5

theta_dim = 3
x_dim = 5

# =============== Load configurations =============== #
# config = json.loads(sys.argv[2])
# num_epochs = config["num_epochs"]
batch_size = 200 # int(config["batch_size"]) 
summary_dim = 128 # int(config["summary_dim"])
embed_hidden_size = 128 # int(config["embed_hidden_size"]) 
embed_num_layers = 3 # int(config["embed_num_layers"])


def main(task_id):
    # ===== generate training data ===== #
    sample_size = 20000
    obs_size = 500
    theta_r0, x_r0 = gen_ref_table(a1, a2, a3, b1, b2, b3, dim = 5, obs_size = obs_size, sample_size = sample_size)
    x_r0 = x_r0.reshape(sample_size, obs_size, -1)
    print(theta_r0.shape, x_r0.shape)


    # ====== Construct the embedding net ====== #
    single_trial_net = FCEmbedding(
        input_dim = x_dim,
        num_hiddens = embed_hidden_size,
        num_layers = embed_num_layers,
        output_dim = summary_dim,
    )
    embedding_net = PermutationInvariantEmbedding(
        single_trial_net,
        trial_net_output_dim=summary_dim
    )

    # Use a normalizing flow as the density estimator
    density_estimator = posterior_nn("maf", embedding_net=embedding_net)


    # ====== Train the model ====== #
    inference = NPE(density_estimator=density_estimator)

    inference.append_simulations(
        theta_r0.to(device),
        x_r0.to(device),
        # exclude_invalid_x=False,
    ).train(
        training_batch_size=batch_size,
        validation_fraction=0.1, 
        show_train_summary=True
        )
    # posterior = inference.build_posterior()

    print(inference._summary)


    # Save the NPE model
    save_dir_nn = Path(f"NPE_embed_model")
    save_dir_nn.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save(inference._neural_net.state_dict(), save_dir_nn / f"npe_net_weights_task{task_id}.pth")



    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)