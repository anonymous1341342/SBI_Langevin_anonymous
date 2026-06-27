from utils_queuing_nmodel import *
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
sample_size = 20000 
batch_size = 500 
hidden_size = 64


# config = json.loads(sys.argv[2])
num_layers = 2 # int(config["num_layers"])
learning_rate = 1e-3 # float(config["learning_rate"])
num_epochs = 3000 # int(config["num_epochs"])
early_stop_patience = 30 # int(config["early_stop_patience"])
sched = True # config["sched"] == "True"


def main(task_id):
    start_time = time.time()
    # ======= Generate Training Data ======== #

    path_theta = Path(f'ref_nmodel/theta_r0_task{task_id}.npy')
    path_x = Path(f'ref_nmodel/x_r0_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists

    if not (path_theta.exists() and path_x.exists()):
        theta_r0, x_r0 = gen_ref_table(a1, a2, a3, b1, b2, b3, dim = 5, sample_size = sample_size)

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



    # Dataloader
    dataset = TensorDataset(train_theta_r0, train_x_r0, prop_score_r0)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    val_dataset = TensorDataset(val_theta_r0, val_x_r0, val_prop_score_r0)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)


    # ===== Training ===== #
    model = Tanh_nmodel_LikeScoreMatchingNN(theta_dim, x_dim, obs_size, hidden_size, num_layers)  
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
    scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)

    if sched:
        sched_patience = 30 # int(config["sched_patience"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)

    path_val_loss_all_dim, path_loss_all_dim = train(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience)


    save_dir = Path(f"nmodel_init")
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
    task_id = int(sys.argv[1])
    main(task_id)