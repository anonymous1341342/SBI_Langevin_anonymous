import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
from utils_queuing_nmodel import *
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
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
obs_size = 500

# =============== Load configurations =============== #

# config = json.loads(sys.argv[2])
# num_epochs = config["num_epochs"]
batch_size = 40 # int(config["batch_size"]) 
summary_dim = 128 # int(config["summary_dim"]) 
embed_hidden_size = 128 # int(config["embed_hidden_size"]) 
embed_num_layers = 3 # int(config["embed_num_layers"]) 

num_rounds = 5 # int(config["num_rounds"]) 
total_sample_size = 20000 # int(config["total_sample_size"]) 

def main(task_id):
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

    # ====== Initialize the model ====== #
    prior = BoxUniform(low=torch.tensor([a1, a2, a3], device=device), high=torch.tensor([b1, b2, b3], device=device))
    inference = NPE(density_estimator=density_estimator, prior=prior, device=device)


    def simulator(theta):
        theta = theta.reshape(-1, theta_dim) # ensure the shape is (sample_size, theta_dim)
        sample_size = theta.shape[0]
        dim = 5

        theta1 = theta[:, 0].cpu().numpy()
        theta2 = theta[:, 1].cpu().numpy()
        theta3 = theta[:, 2].cpu().numpy()

        x_stretched = np.zeros((sample_size, obs_size * dim))
        for j in range(obs_size):
            # generate w and u
            w = np.zeros((sample_size, dim))
            u = np.zeros((sample_size, dim))
            for i in range(dim):
                w[:, i] = np.random.exponential(scale = 1.0/theta3, size = sample_size) # scale = inverse rate
                u[:, i] = np.random.uniform(low = theta1, high = theta1 + theta2, size = sample_size)
                
            # use w and u to calculate x
            x = np.zeros((sample_size, dim))
            x[:, 0] = u[:, 0] + w[:, 0]
            for k in range(1, dim):
                # tmp = np.sum(w[:, :(k+1)], axis = 1) - np.sum(x[:, :k], axis = 1) # take k+1, as right boundary is not included
                x[:, k] = u[:, k] + np.maximum(0, np.sum(w[:, :(k+1)], axis = 1) - np.sum(x[:, :k], axis = 1))
            x_stretched[:, (j * dim):((j+1) * dim)] = x # stack all the observed x

        return torch.tensor(x_stretched, dtype = torch.float32).reshape(sample_size, obs_size, -1).to(device)



    x_obs = pd.read_csv(f"data_obs/x_obs_task{task_id}.csv")
    x_obs = torch.tensor(x_obs.values, dtype=torch.float32).contiguous()

    # The embedding net expects shape (batch, num_trials, x_dim).
    x_o = x_obs.reshape(1, obs_size, x_dim)


    # Ensure compliance with sbi's requirements.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)

    posteriors = []
    proposal = prior


    # total_sample_size = 20000 
    for _ in range(num_rounds):
        # theta, x = simulate_for_sbi(simulator, proposal, num_simulations=int(total_sample_size / num_rounds))
        num_simulations=int(total_sample_size / num_rounds)
        print(f"Using {num_simulations} simulations for this round.")
        theta = proposal.sample((num_simulations,))
        x = simulator(theta)

        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train(
        training_batch_size=batch_size,
        validation_fraction=0.1, 
        show_train_summary=True
        )
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)


    # ======= Sampling ======= #
    for round_idx, posterior in enumerate(posteriors):
        samples = posterior.sample((10000,), x=x_o.to(device), show_progress_bars=False)
        # create folder if not exists, and save as .npy
        os.makedirs(f"res_NPE_embed_sequential_{num_rounds}rounds_{total_sample_size}budget", exist_ok=True)
        np.save(f"res_NPE_embed_sequential_{num_rounds}rounds_{total_sample_size}budget/samples_task{task_id}_round{round_idx}.npy", samples.cpu().numpy())



    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')



if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)