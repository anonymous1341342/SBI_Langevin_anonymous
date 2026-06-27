from utils_sm import *
from utils_SDEMEM_realdata import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path
from scipy.stats import truncnorm

# ==== Load configurations ==== #
hidden_size = 128 # config["hidden_size"]
num_layers = 3 # config["num_layers"]
num_epochs = 300 # config["num_epochs"]
learning_rate = 1e-4 # config["learning_rate"]
batch_size = 1000 #  int(config["batch_size"])
ref_size = int(1e7) # int(config["ref_size"])
sm_rd = 1 

early_stop_patience = 10 # config["early_stop_patience"]
sched = False


# ===== Setting for real data ===== #
T = 30
theta_dim = 12
x_dim = 180 


def main(task_id):
    start_time = time.time()
    ##########################
    # Generate Training Data #
    ##########################
    # Load SW data
    theta_SW1 = np.load("res_SW1/theta_SW1.npy")
    loss_SW1 = np.load("res_SW1/final_loss.npy")


    nan_idx = np.isnan(theta_SW1).any(axis=1)
    theta_SW1 = theta_SW1[~nan_idx]
    loss_SW1 = loss_SW1[~nan_idx]

    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)[:100]
    print(f"theta_SW1.shape = {theta_SW1.shape}")


    prop_mean = theta_SW1.mean(dim = 0, keepdims = True)
    prop_std = theta_SW1.std(dim = 0, keepdims = True) 

    # inflate the proposal std
    prop_std *= 2

    # the previous prop_std is too small for these two dimensions
    prop_std[0, 0] *= 3
    prop_std[0, 6] *= 3


    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")

    # generate reference table
    path_theta = Path(f'ref_S/sm_round{sm_rd}/theta_r0_task{task_id}.npy')
    path_x = Path(f'ref_S/sm_round{sm_rd}/x_r0_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists

    if not (path_theta.exists() and path_x.exists()):
        theta_r0 = torch.empty(ref_size, theta_dim)
        x_r0 = torch.empty(ref_size, x_dim)

        step = max(1, int(ref_size / 5))
        print(f"step = {step}")
        kept = 0

        for start in range(0, ref_size, step):
            end = min(start + step, ref_size)
            current_n = end - start

            theta_part = prop_mean + prop_std * torch.randn(current_n, theta_dim)
            x_part = gen_x_given_theta(theta_part.to(device), T=T).cpu()

            valid_mask = torch.isfinite(x_part).all(dim=1)
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                continue

            theta_r0[kept:kept + num_valid].copy_(theta_part[valid_mask])
            x_r0[kept:kept + num_valid].copy_(x_part[valid_mask])
            kept += num_valid

        theta_r0 = theta_r0[:kept]
        x_r0 = x_r0[:kept]

        print(f"generated reference table with shape theta: {theta_r0.shape}, x: {x_r0.shape}")
        np.save(path_theta, theta_r0.numpy())
        np.save(path_x, x_r0.numpy())


    else:
        theta_r0 = torch.from_numpy(np.load(path_theta, mmap_mode="r")[:ref_size].copy())
        x_r0 = torch.from_numpy(np.load(path_x, mmap_mode="r")[:ref_size].copy())
        print(f"dtype = {theta_r0.dtype}, {x_r0.dtype}")


    # Split training and validation
    N_tr = int(0.9 * ref_size)
    theta_tr, theta_val = theta_r0[:N_tr], theta_r0[N_tr:]
    x_tr, x_val = x_r0[:N_tr], x_r0[N_tr:]


    # Standardize data
    theta_tr = (theta_tr - prop_mean) / prop_std
    theta_val = (theta_val - prop_mean) / prop_std

    mean_x, std_x = x_tr.mean(dim = 0, keepdims = True), x_tr.std(dim = 0, keepdims = True).clamp_min(1e-8)

    x_tr -= mean_x
    x_tr /= std_x

    x_val -= mean_x
    x_val /= std_x

    # prop score
    prop_score_tr = -theta_tr
    prop_score_val = -theta_val

    print(f"Training data number = {theta_tr.shape[0]}, Validation data number = {theta_val.shape[0]}")

    #####################################################
    #          Determine the weight function            #
    #####################################################
    def g(theta, x):
        return torch.ones_like(theta)

    def g1(theta, x):
        return torch.zeros_like(theta)

    ##########################
    # Prepare the Dataloader #
    ##########################
    train_set = TensorDataset(theta_tr, x_tr, prop_score_tr)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

    val_set = TensorDataset(theta_val, x_val, prop_score_val)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)


    ############
    # Training #
    ############

    model = ELU_single_LikeScoreMatchingNN_sparse(theta_dim, x_dim, hidden_size, num_layers)
        
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

    scheduler = None

    # train the model
    bias_lastlayer, path_val_loss_all_dim, path_loss_all_dim = train_deb_freeze(model, optimizer, train_loader, val_loader, g, g1, num_epochs, scheduler, early_stop_patience)

    save_dir = Path(f"model_single_noscale/sm_round{sm_rd}")
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bias_lastlayer': bias_lastlayer,
        'path_val_loss_all_dim': path_val_loss_all_dim,
        'path_loss_all_dim': path_loss_all_dim,
        'mean_x': mean_x,
        'std_x': std_x
    }, save_dir / f"checkpoint_task{task_id}.pth")


    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/3600, 2)} hours')


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)