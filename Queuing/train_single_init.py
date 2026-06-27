from utils_queuing_single import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd


# ===== Setting ===== #
obs_size = 500
theta_dim = 3
x_dim = 5

a1 = 0.0
b1 = 10.0

a2 = 0.0
b2 = 10.0

a3 = 0.01 #
b3 = 0.5



# ======= Configurations ======= #
# task_id = sys.argv[1]

hidden_size = 64 # config["hidden_size"]
num_layers = 3 # config["num_layers"]
num_epochs = 200 # config["num_epochs"]
learning_rate = 1e-3 # config["learning_rate"]
batch_size = 500 # int(config["batch_size"])
sample_size = int(2e5) # int(config["sample_size"])
model_noise = 0.25 # config["model_noise"]

early_stop_patience = 10 # config["early_stop_patience"]
sched = True # config["sched"] == "True"
sched_patience = 5 # int(config["sched_patience"])


print(f"model_noise = {model_noise}")



def main(task_id):
    start_time = time.time()
    ##########################
    # Generate Training Data #
    ##########################
    path_theta = Path(f'ref_S/mn{model_noise}/theta_r0_task{task_id}.npy')
    path_x = Path(f'ref_S/mn{model_noise}/x_r0_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists

    if not (path_theta.exists() and path_x.exists()):
        theta_r0, x_r0 =gen_ref_table_distinct_theta(a1, a2, a3, b1, b2, b3, dim = 5, sample_size = sample_size)
        x_r0 = x_r0 + model_noise * torch.randn(x_r0.shape)

        print(f"generated reference table with shape theta: {theta_r0.shape}, x(with noise injected): {x_r0.shape}")
        np.save(path_theta, theta_r0.numpy())
        np.save(path_x, x_r0.numpy())
    else:
        theta_r0 = torch.from_numpy(np.load(path_theta))
        x_r0 = torch.from_numpy(np.load(path_x))
        print(f"dtype = {theta_r0.dtype}, {x_r0.dtype}")

    tr_size = int(0.9 * sample_size)
    train_theta_r0 = theta_r0[:tr_size]
    train_x_r0 = x_r0[:tr_size]
    val_theta_r0 = theta_r0[tr_size:]
    val_x_r0 = x_r0[tr_size:]

    prop_score_r0 = torch.zeros(train_theta_r0.shape)
    val_prop_score_r0 = torch.zeros(val_theta_r0.shape) # validation set

    ##########################
    # Prepare the Dataloader #
    ##########################
    train_dataset = TensorDataset(train_theta_r0, train_x_r0, prop_score_r0)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataset = TensorDataset(val_theta_r0, val_x_r0, val_prop_score_r0)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)


    ############
    # Training #
    ############

    model = ELU_single_LikeScoreMatchingNN(theta_dim, x_dim, hidden_size, num_layers)  
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)

    bias_lastlayer, path_val_loss_all_dim, path_loss_all_dim = train_deb(model, optimizer, train_dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience)


    save_dir = Path(f"model_single_init/mn{model_noise}")
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bias_lastlayer': bias_lastlayer,
        'path_val_loss_all_dim': path_val_loss_all_dim,
        'path_loss_all_dim': path_loss_all_dim,
    }, save_dir / f"checkpoint_task{task_id}.pth")


    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/3600, 2)} hours')



if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)




