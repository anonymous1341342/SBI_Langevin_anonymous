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
    # with torch.no_grad(): 
    #     model.layers[-1].bias -= (total_bias / len(dataloader)).to(device) 

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






# ==== Load configurations ==== #
# task_id = int(sys.argv[1]) # is also the obs_id


hidden_size = 64 # config["hidden_size"]
num_layers = 3 # config["num_layers"]
num_epochs = 50 # config["num_epochs"]
learning_rate = 3e-5 # config["learning_rate"]
batch_size = 500 # int(config["batch_size"])
sample_size = int(2e5) # int(config["sample_size"])

extra_ref_size = int(1e4) # int(config["extra_ref_size"])
extra_obs_size = 500 # int(config["extra_obs_size"])
batch_size_extra = 20 # int(config["batch_size_extra"])


early_stop_patience = 10 # config["early_stop_patience"]
sched = False

lam_fisher = 1e-3 # config["lam_fisher"]
print(f"lam_fisher = {lam_fisher}")


model_noise = 0.25 # config["model_noise"]
print(f"model_noise = {model_noise}")



def main(task_id):
    # ============= ref_S ============== #
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


    train_dataset = TensorDataset(train_theta_r0, train_x_r0, prop_score_r0)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataset = TensorDataset(val_theta_r0, val_x_r0, val_prop_score_r0)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)



    # ============= ref_R ============== #
    path_theta = Path(f'ref_R/mn{model_noise}/theta_r0_extra_task{task_id}.npy')
    path_x = Path(f'ref_R/mn{model_noise}/x_r0_extra_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists


    if not (path_theta.exists() and path_x.exists()):
        theta_r0_extra, x_r0_extra = gen_ref_table(a1, a2, a3, b1, b2, b3, dim = 5, obs_size = extra_obs_size, sample_size = extra_ref_size)
        x_r0_extra = x_r0_extra + model_noise * torch.randn(x_r0_extra.shape)

        print(f"generated reference table with shape theta: {theta_r0_extra.shape}, x(with noise): {x_r0_extra.shape}")
        np.save(path_theta, theta_r0_extra.numpy())
        np.save(path_x, x_r0_extra.numpy())
    else:
        theta_r0_extra = torch.from_numpy(np.load(path_theta))
        x_r0_extra = torch.from_numpy(np.load(path_x))
        print(f"dtype = {theta_r0_extra.dtype}, {x_r0_extra.dtype}")

    tr_size = int(0.9 * extra_ref_size)
    train_theta_r0_extra = theta_r0_extra[:tr_size]
    train_x_r0_extra = x_r0_extra[:tr_size]
    val_theta_r0_extra = theta_r0_extra[tr_size:]
    val_x_r0_extra = x_r0_extra[tr_size:]



    extra_train_set = TensorDataset(train_theta_r0_extra, train_x_r0_extra)
    extra_train_loader = DataLoader(extra_train_set, batch_size = batch_size_extra, shuffle=True)
    extra_val_set = TensorDataset(val_theta_r0_extra, val_x_r0_extra)
    extra_val_loader = DataLoader(extra_val_set, batch_size = batch_size_extra, shuffle=False)



    ################################
    #          Training            #
    ################################
    # Create model and optimizer
    model = ELU_single_LikeScoreMatchingNN(theta_dim, x_dim, hidden_size, num_layers)

    # start from the trained model without curvature penalty
    checkpoint = torch.load(f'model_single_init/mn{model_noise}/checkpoint_task{task_id}.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # CHECK LOSS
    print("Loss of the initial model")
    val_sm_loss_init = check_loss(model, val_loader, extra_val_loader, lam_fisher, g)
    print("\n")

    # continue training from the initialized model
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
    scheduler = None

    bias_lastlayer = train_fisher(model, optimizer, train_loader, val_loader, extra_train_loader, extra_val_loader, lam_fisher, g, g1, num_epochs, scheduler)
    print("Loss after fisher training")
    val_sm_loss_fisher = check_loss(model, val_loader, extra_val_loader, lam_fisher, g)
    print("\n")



    save_dir = Path(f"model_single_fisher/mn{model_noise}")
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    if val_sm_loss_fisher > val_sm_loss_init * 0.99: # if no significant improvement, then use the initial model
        print("Returning to the initial model")
        checkpoint = torch.load(f'model_single_init/mn{model_noise}/checkpoint_task{task_id}.pth', map_location=device, weights_only=False)
        torch.save(checkpoint, save_dir / f'checkpoint_task{task_id}.pth')
    else: # use the penalized model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bias_lastlayer': bias_lastlayer,
        }, save_dir / f'checkpoint_task{task_id}.pth')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)