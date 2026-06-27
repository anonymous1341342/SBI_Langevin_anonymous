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
num_epochs = 12 # config["num_epochs"]
learning_rate = 5e-5 # config["learning_rate"]
batch_size = 1000 # int(config["batch_size"])
ref_size = int(1e7) # int(config["ref_size"])
sm_rd = 1 

early_stop_patience = 10 # config["early_stop_patience"]
sched = True



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
        print("should have data here")
    else:
        theta_r0 = torch.from_numpy(np.load(path_theta))
        x_r0 = torch.from_numpy(np.load(path_x))
        print(f"dtype = {theta_r0.dtype}, {x_r0.dtype}")


    # Split training and validation
    N_tr = int(0.9 * ref_size)
    theta_tr, theta_val = theta_r0[:N_tr], theta_r0[N_tr:]
    x_tr, x_val = x_r0[:N_tr], x_r0[N_tr:]


    # Standardize data
    theta_tr -= prop_mean
    theta_tr /= prop_std

    theta_val -= prop_mean
    theta_val /= prop_std

    mean_x, std_x = x_tr.mean(dim = 0, keepdims = True), x_tr.std(dim = 0, keepdims = True).clamp_min(1e-8)

    x_tr -= mean_x
    x_tr /= std_x

    x_val -= mean_x
    x_val /= std_x

    # prop score
    prop_score_tr = -theta_tr
    prop_score_val = -theta_val



    # ========== Use the scale of scores obtained from the pretraining to set the weight to adjust the scale difference          
    checkpoint_path = f"model_single_noscale/sm_round{sm_rd}/checkpoint_task{task_id}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    path_val_loss_all_dim = checkpoint['path_val_loss_all_dim']
    path_val_loss_all_dim = np.stack(path_val_loss_all_dim, axis = 0) # shape (num_epochs, theta_dim)
    scale_score = torch.tensor( -path_val_loss_all_dim.min(axis=0), dtype = torch.float32)

    assert torch.all(scale_score > 0), "All dimensions of scale_score should be positive"
    print(f"Using {scale_score} to weight the score matching loss")

    # CHANGED THE WEIGHT HERE
    def g(theta, x):
        return torch.ones_like(theta) / scale_score.to(theta.device)

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

    # load checkpoint and continue to train the model, but with the weight to adjust for scale difference
    # rebuild model and optimizer
    model = ELU_single_LikeScoreMatchingNN_sparse(theta_dim, x_dim, hidden_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # restore saved states
    model.load_state_dict(checkpoint['model_state_dict'])


    scheduler = None
    if sched:
        sched_step_size = 3 # int(config["sched_step_size"])
        sched_gamma = 0.5 # float(config["sched_gamma"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)
        print(f"Using StepLR scheduler, with step_size {sched_step_size} and gamma {sched_gamma}")

    # continue training
    bias_lastlayer, _ = train_deb(model, optimizer, train_loader, val_loader, g, g1, num_epochs, scheduler, early_stop_patience, return_best_model=True)

    save_dir = Path(f"model_single_weighted/sm_round{sm_rd}")
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bias_lastlayer': bias_lastlayer,
        'path_val_loss_all_dim': path_val_loss_all_dim, # this is from the noscale training
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