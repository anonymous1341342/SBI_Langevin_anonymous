from utils_SDEMEM_realdata import *
from utils_sm import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path


# ==== Load configurations ==== #
hidden_size = 128 # config["hidden_size"]
num_layers = 3 # config["num_layers"]
num_epochs = 2 # config["num_epochs"]
learning_rate = 6.25e-6 # config["learning_rate"]
batch_size = 1000 # int(config["batch_size"])
ref_size = int(1e7) # int(config["ref_size"])
sm_rd = 1 

extra_ref_size = int(1e6) # int(config["extra_ref_size"])
extra_obs_size = 40 # int(config["extra_obs_size"])
batch_size_extra = 20 # int(config["batch_size_extra"])

early_stop_patience = 10 # config["early_stop_patience"]
sched = True

lam_fisher = 1e-3 # config["lam_fisher"]



# ===== Setting for real data ===== #
T = 30
theta_dim = 12
x_dim = 180 


def main(task_id):
    def train_fisher(model, optimizer, dataloader, val_dataloader, dataloader_extra, val_dataloader_extra, lam_fisher, g, g1, num_epochs, scheduler, return_best_model = True):
        print(f"Train with penalty lam_fisher = {lam_fisher}")
        model.to(device)
        best_val_sm_loss = float('inf')
        best_model_state = None
        best_optimizer_state = None

        # record training loss and validation loss at each epoch and then plot
        start_time = time.time()
        for epoch in range(num_epochs):
            time1 = time.time()
            model.train() 
            total_loss = 0.0
            total_sm_loss = 0.0
            total_penalty_fisher = 0.0
            
            data_extra_iter = cycle(dataloader_extra)
            valid_batches = 0
            total_sm_loss_alldim = torch.zeros(model.theta_dim).to(device)
            for iter_counter, batch_sample in enumerate(dataloader):
                batch_sample_extra = next(data_extra_iter)
                optimizer.zero_grad()
                batch_theta, batch_x, batch_prop_score = batch_sample
                batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
                
                batch_theta_extra, batch_x_extra = batch_sample_extra
                batch_theta_extra, batch_x_extra = batch_theta_extra.to(device), batch_x_extra.to(device)

                sm_loss, bias, sm_loss_alldim = Like_score_loss_deb(model, batch_theta, batch_x, batch_prop_score, g, g1)
                penalty_fisher = weighted_Fisher_penalty(model, batch_theta_extra, batch_x_extra, g)                       
                loss = sm_loss + lam_fisher * penalty_fisher
                    
                if torch.isnan(loss):
                    print(f"[WARNING] NaN detected, skipping this minibatch")
                    continue
                valid_batches += 1
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_sm_loss += sm_loss.item()
                total_sm_loss_alldim += sm_loss_alldim.detach()
                total_penalty_fisher += penalty_fisher.item()

            model.eval()
            val_total_loss = 0.0
            val_total_sm_loss = 0.0
            val_total_penalty_fisher = 0.0

            val_data_extra_iter = cycle(val_dataloader_extra)
            val_valid_batches = 0
            val_total_sm_loss_alldim = torch.zeros(model.theta_dim).to(device)
            for val_batch_sample in val_dataloader:
                val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_sample
                val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_theta.to(device), val_batch_x.to(device), val_batch_prop_score.to(device)

                val_batch_sample_extra = next(val_data_extra_iter)
                val_batch_theta_extra, val_batch_x_extra = val_batch_sample_extra
                val_batch_theta_extra, val_batch_x_extra = val_batch_theta_extra.to(device), val_batch_x_extra.to(device)

                val_sm_loss, val_bias, val_sm_loss_alldim = Like_score_loss_deb(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
                val_penalty_fisher = weighted_Fisher_penalty(model, val_batch_theta_extra, val_batch_x_extra, g)                       
                val_loss = val_sm_loss + lam_fisher * val_penalty_fisher
                
                if torch.isnan(val_loss):
                    print(f"[WARNING] NaN detected, skipping this minibatch")
                    continue
                val_valid_batches += 1
                
                val_total_loss += val_loss.item()    
                val_total_sm_loss += val_sm_loss.item()
                val_total_sm_loss_alldim += val_sm_loss_alldim.detach()
                val_total_penalty_fisher += val_penalty_fisher.item()

            avg_val_sm_loss = val_total_sm_loss / val_valid_batches
            if avg_val_sm_loss < best_val_sm_loss:
                best_epoch = epoch + 1
                best_val_sm_loss = avg_val_sm_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())


            time2 = time.time()
            if epoch % 1 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} | Training Loss (Total, SM, pen_fisher): ({total_loss / valid_batches:.3f}, {total_sm_loss / valid_batches:.3f}, {total_penalty_fisher / valid_batches:.3f}) | Validation Loss (Total, SM, pen_fisher): ({val_total_loss / val_valid_batches:.3f}, {val_total_sm_loss / val_valid_batches:.3f}, {val_total_penalty_fisher / val_valid_batches:.3f}). Time: {(time2 - time1):.2f} seconds')
                print(f'    Training SM Loss (alldim): {np.round(total_sm_loss_alldim.cpu().numpy() / valid_batches, 3)} | Validation SM Loss (alldim): {np.round(val_total_sm_loss_alldim.cpu().numpy() / val_valid_batches, 3)}')

            if scheduler is not None:
                old_lr = optimizer.param_groups[0]["lr"]

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_total_loss / val_valid_batches) # use the penalized loss here
                else:
                    scheduler.step()

                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != old_lr:
                    print(f"Epoch {epoch+1}: reducing learning rate to {new_lr:.2e}")


        # Load best model state after training
        if return_best_model and best_model_state is not None:
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_optimizer_state)
            print(f"Return the best model at epoch {best_epoch}, with Validation sm Loss: {best_val_sm_loss:.3f}")

        
        # output the final model, we just need to minus the bias
        # we calculate the bias using the whole dataset
        total_bias = 0.0 # is actually a vector of the same dimension as theta
        for batch_sample in dataloader:
            batch_theta, batch_x, batch_prop_score = batch_sample
            batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
            loss, bias, _ = Like_score_loss_deb(model, batch_theta, batch_x, batch_prop_score, g, g1)
            total_bias += bias.detach() 

        bias_lastlayer = total_bias / len(dataloader)
        end_time = time.time()
        total_duration = end_time - start_time
        print(f'Total training time: {total_duration/60:.2f} minutes')
        return bias_lastlayer



    def check_loss(model, val_dataloader, val_dataloader_extra, lam_fisher, g):
        model.eval()
        val_total_loss = 0.0
        val_total_sm_loss = 0.0
        val_total_penalty_fisher = 0.0
        val_total_scale = 0.0
        
        val_data_extra_iter = cycle(val_dataloader_extra)
        val_valid_batches = 0
        for val_batch_sample in val_dataloader:
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_sample
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_theta.to(device), val_batch_x.to(device), val_batch_prop_score.to(device)
            val_batch_sample_extra = next(val_data_extra_iter)
            val_batch_theta_extra, val_batch_x_extra = val_batch_sample_extra
            val_batch_theta_extra, val_batch_x_extra = val_batch_theta_extra.to(device), val_batch_x_extra.to(device)
            val_sm_loss, val_bias, _ = Like_score_loss_deb(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
            val_penalty_fisher = weighted_Fisher_penalty(model, val_batch_theta_extra, val_batch_x_extra, g)                       
            val_loss = val_sm_loss + lam_fisher * val_penalty_fisher
            val_scale_ssT = cal_weighted_ssT(model, val_batch_theta_extra, val_batch_x_extra, g)
            
            if torch.isnan(val_loss):
                print(f"[WARNING] NaN detected, skipping this minibatch")
                continue
            val_valid_batches += 1
            
            val_total_loss += val_loss.item()    
            val_total_sm_loss += val_sm_loss.item()
            val_total_penalty_fisher += val_penalty_fisher.item()
            val_total_scale += val_scale_ssT.item()
        
        print(f'Validation Loss (Total, SM, pen_fisher): ({val_total_loss / val_valid_batches:.3f}, {val_total_sm_loss / val_valid_batches:.3f}, {val_total_penalty_fisher / val_valid_batches:.3f})')

        print(f'scale E[||EssT||_F^2] = {val_total_scale / val_valid_batches:.3f}')
        return val_total_sm_loss / val_valid_batches # return the score matching loss




    start_time = time.time()
    #################
    # Training Data #
    #################
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


    # ===== ref_S ===== #
    path_theta = Path(f'ref_S/sm_round{sm_rd}/theta_r0_task{task_id}.npy')
    path_x = Path(f'ref_S/sm_round{sm_rd}/x_r0_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists

    if not (path_theta.exists() and path_x.exists()):
        theta_r0 = torch.empty(ref_size, theta_dim)
        x_r0 = torch.empty(ref_size, x_dim)

        step = max(1, int(ref_size / 2)) # max(1, int(ref_size / 10))
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



    # ===== ref_R ===== #
    path_theta = Path(f'ref_R/sm_round{sm_rd}/theta_r0_extra_task{task_id}.npy')
    path_x = Path(f'ref_R/sm_round{sm_rd}/x_r0_extra_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists

    if not (path_theta.exists() and path_x.exists()):
        theta_r0_extra = torch.empty(extra_ref_size, theta_dim)
        x_r0_extra = torch.empty(extra_ref_size, extra_obs_size * x_dim)

        step = max(1, int(extra_ref_size / 2)) # max(1, int(extra_ref_size / 10))
        kept = 0

        for start in range(0, extra_ref_size, step):
            end = min(start + step, extra_ref_size)
            current_n = end - start

            theta_part = prop_mean + prop_std * torch.randn(current_n, theta_dim)
            x_part = torch.empty(current_n, extra_obs_size * x_dim)

            for i in range(extra_obs_size):
                x_part[:, i * x_dim:(i + 1) * x_dim] = gen_x_given_theta(theta_part.to(device), T=T).cpu()

            valid_mask = torch.isfinite(x_part).all(dim=1)
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                continue

            theta_r0_extra[kept:kept + num_valid].copy_(theta_part[valid_mask])
            x_r0_extra[kept:kept + num_valid].copy_(x_part[valid_mask])
            kept += num_valid

        theta_r0_extra = theta_r0_extra[:kept]
        x_r0_extra = x_r0_extra[:kept]

        print(f"generated reference table with shape theta: {theta_r0_extra.shape}, x: {x_r0_extra.shape}")

        np.save(path_theta, theta_r0_extra.numpy())
        np.save(path_x, x_r0_extra.numpy())
    else:
        theta_r0_extra = torch.from_numpy(np.load(path_theta))
        x_r0_extra = torch.from_numpy(np.load(path_x))
        print(f"extra data dtype = {theta_r0_extra.dtype}, {x_r0_extra.dtype}")


    # standardize theta
    theta_r0_extra = (theta_r0_extra - prop_mean) / prop_std
    x_r0_extra -= mean_x.repeat(1, extra_obs_size)
    x_r0_extra /= std_x.repeat(1, extra_obs_size)


    # split train and validation
    N_tr = int(0.9 * extra_ref_size)
    train_theta_extra = theta_r0_extra[:N_tr]
    train_x_extra = x_r0_extra[:N_tr]

    val_theta_extra = theta_r0_extra[N_tr:]
    val_x_extra = x_r0_extra[N_tr:]

    print(f"Number of training data for curvature penalty = {train_theta_extra.shape[0]}")
    print(f"Number of validation data for curvature penalty = {val_theta_extra.shape[0]}")

    #####################################################
    #          Determine the weight function            #
    #####################################################
    # Load the SingleModel
    checkpoint_path = f"model_single_weighted/sm_round{sm_rd}/checkpoint_task{task_id}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    path_val_loss_all_dim = checkpoint['path_val_loss_all_dim']
    path_val_loss_all_dim = np.stack(path_val_loss_all_dim, axis = 0) # shape (num_epochs, theta_dim)
    scale_score = torch.tensor( -path_val_loss_all_dim.min(axis=0), dtype = torch.float32)

    print(f"using scale_score = {scale_score}")

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

    extra_train_set = TensorDataset(train_theta_extra, train_x_extra)
    extra_train_loader = DataLoader(extra_train_set, batch_size = batch_size_extra, shuffle=True)
    extra_val_set = TensorDataset(val_theta_extra, val_x_extra)
    extra_val_loader = DataLoader(extra_val_set, batch_size = batch_size_extra, shuffle=False)



    ################################
    #          Training            #
    ################################
    # Create model and optimizer
    model = ELU_single_LikeScoreMatchingNN_sparse(theta_dim, x_dim, hidden_size, num_layers)

    # start from the trained model without curvature penalty
    checkpoint = torch.load(f'model_single_weighted/sm_round{sm_rd}/checkpoint_task{task_id}.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # CHECK LOSS
    print("Loss of the initial model")
    val_sm_loss_init = check_loss(model, val_loader, extra_val_loader, lam_fisher, g)
    print("\n")

    # continue training from the initialized model
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

    scheduler = None
    if sched:
        sched_step_size = 2 # int(config["sched_step_size"])
        sched_gamma = 0.5 # float(config["sched_gamma"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)
        print(f"Using StepLR scheduler, with step_size {sched_step_size} and gamma {sched_gamma}")


    bias_lastlayer = train_fisher(model, optimizer, train_loader, val_loader, extra_train_loader, extra_val_loader, lam_fisher, g, g1, num_epochs, scheduler)
    print("Loss after fisher training")
    val_sm_loss_fisher = check_loss(model, val_loader, extra_val_loader, lam_fisher, g)
    print("\n")


    save_dir = Path(f"scaled_fishermodel_weighted/sm_round{sm_rd}")
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    if val_sm_loss_fisher > val_sm_loss_init * 0.99: # if no significant improvement, then use the initial model
        print("Returning to the initial model")
        checkpoint = torch.load(f'model_single_weighted/sm_round{sm_rd}/checkpoint_task{task_id}.pth', map_location=device, weights_only=False)
        torch.save(checkpoint, save_dir / f'checkpoint_task{task_id}.pth')
    else: # use the penalized model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bias_lastlayer': bias_lastlayer,
            'path_val_loss_all_dim': path_val_loss_all_dim,
            'mean_x': mean_x,
            'std_x': std_x
        }, save_dir / f'checkpoint_task{task_id}.pth')

    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/3600, 2)} hours')


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)