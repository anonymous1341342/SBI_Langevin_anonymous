from utils_monoBP_single import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import sys
import json
from torch.utils.data import Dataset



# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0

M = 10


hidden_size = 64 # config["hidden_size"]
num_layers = 3 # config["num_layers"]
num_epochs = 50 # config["num_epochs"]
learning_rate = 1e-5 # config["learning_rate"]
batch_size = int(1e3) # int(config["batch_size"])
training_size = int(1e6) # int(config["training_size"])

extra_sample_size = 50000 # int(config["extra_sample_size"])
extra_obs_size = 1000 # int(config["extra_obs_size"])
batch_size_extra = 40 # int(config["batch_size_extra"])

lam_fisher = 1e-3
sched = False



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.version.cuda) 
print(torch.cuda.is_available()) 



def main(task_id):
    start_time = time.time()


    def train_fisher(model, optimizer, dataloader, val_dataloader, dataloader_extra, val_dataloader_extra, lam_fisher, g, g1, num_epochs, scheduler):
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
            for iter_counter, batch_sample in enumerate(dataloader):
                batch_sample_extra = next(data_extra_iter)
                # print(f"Extra_data_head: {batch_sample_extra[:2]}") # verify the usage is correct
                optimizer.zero_grad()
                batch_theta, batch_x, batch_prop_score = batch_sample
                batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
                
                batch_theta_extra, batch_x_extra = batch_sample_extra
                batch_theta_extra, batch_x_extra = batch_theta_extra.to(device), batch_x_extra.to(device)

                sm_loss, bias = Like_score_loss(model, batch_theta, batch_x, batch_prop_score, g, g1)
                penalty_fisher = Fisher_penalty(model, batch_theta_extra, batch_x_extra)                       
                loss = sm_loss + lam_fisher * penalty_fisher
                    
                if torch.isnan(loss):
                    print(f"[WARNING] NaN detected, skipping this minibatch")
                    continue
                valid_batches += 1
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_sm_loss += sm_loss.item()
                total_penalty_fisher += penalty_fisher.item()

            model.eval()
            val_total_loss = 0.0
            val_total_sm_loss = 0.0
            val_total_penalty_fisher = 0.0

            val_data_extra_iter = cycle(val_dataloader_extra)
            val_valid_batches = 0
            for val_batch_sample in val_dataloader:
                val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_sample
                val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_theta.to(device), val_batch_x.to(device), val_batch_prop_score.to(device)

                val_batch_sample_extra = next(val_data_extra_iter)
                val_batch_theta_extra, val_batch_x_extra = val_batch_sample_extra
                val_batch_theta_extra, val_batch_x_extra = val_batch_theta_extra.to(device), val_batch_x_extra.to(device)

                val_sm_loss, val_bias = Like_score_loss(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
                val_penalty_fisher = Fisher_penalty(model, val_batch_theta_extra, val_batch_x_extra)                       
                val_loss = val_sm_loss + lam_fisher * val_penalty_fisher
                
                if torch.isnan(val_loss):
                    print(f"[WARNING] NaN detected, skipping this minibatch")
                    continue
                val_valid_batches += 1
                
                val_total_loss += val_loss.item()    
                val_total_sm_loss += val_sm_loss.item()
                val_total_penalty_fisher += val_penalty_fisher.item()

            avg_val_sm_loss = val_total_sm_loss / val_valid_batches
            if avg_val_sm_loss < best_val_sm_loss:
                best_epoch = epoch + 1
                best_val_sm_loss = avg_val_sm_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())

            if scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_total_loss / val_valid_batches) # use the penalized loss here
                new_lr = scheduler.get_last_lr()[0]
                if new_lr != old_lr:
                    print(f"Epoch {epoch+1}: reducing learning rate to {new_lr:.2e}")
            
            time2 = time.time()
            if epoch % 1 == 0 or epoch == num_epochs:
                print(f'Epoch {epoch+1}/{num_epochs} | Training Loss (Total, SM, pen_fisher): ({total_loss / valid_batches:.3f}, {total_sm_loss / valid_batches:.3f}, {total_penalty_fisher / valid_batches:.3f}) | Validation Loss (Total, SM, pen_fisher): ({val_total_loss / val_valid_batches:.3f}, {val_total_sm_loss / val_valid_batches:.3f}, {val_total_penalty_fisher / val_valid_batches:.3f}). Time: {(time2 - time1):.2f} seconds')


        # Load best model state after training
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_optimizer_state)
            print(f"Return the best model at epoch {best_epoch}, with Validation sm Loss: {best_val_sm_loss:.3f}")

        
        # output the final model, we just need to minus the bias
        # we calculate the bias using the whole dataset
        total_bias = 0.0 # is actually a vector of the same dimension as theta
        for batch_sample in dataloader:
            batch_theta, batch_x, batch_prop_score = batch_sample
            batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
            loss, bias = Like_score_loss(model, batch_theta, batch_x, batch_prop_score, g, g1)
            total_bias += bias.detach()


        bias_lastlayer = total_bias / len(dataloader)
        end_time = time.time()
        total_duration = end_time - start_time
        print(f'Total training time: {total_duration/60:.2f} minutes')
        return bias_lastlayer



    def check_loss(model, val_dataloader, val_dataloader_extra, lam_fisher):
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
            val_sm_loss, val_bias = Like_score_loss(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
            val_penalty_fisher = Fisher_penalty(model, val_batch_theta_extra, val_batch_x_extra)                       
            val_loss = val_sm_loss + lam_fisher * val_penalty_fisher
            val_scale_ssT = cal_ssT(model, val_batch_theta_extra, val_batch_x_extra)
            
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




    #############################################################################################
    #        Read previously generated data: data_obs and the SW preconditioned samples         #
    #############################################################################################
    sigma = 0.1 
    obs_size = 1000

    data_obs = pd.read_csv(f"data_obs/data_obs_task{task_id}.csv")
    data_obs = torch.tensor(data_obs.values, dtype = torch.float32).contiguous().to(device)

    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()

    ##################################
    #          Training Data         #
    ##################################

    lower = torch.zeros(M + 1) # .to(device)
    lower[0] = a0
    lower[1:] = a
    upper = torch.zeros(M + 1) # .to(device)
    upper[0] = b0
    upper[1:] = b

    actual_inf_rate = torch.ones(M + 1)
    actual_inf_rate[-2:] = 2

    inf_rate = torch.zeros(M + 1)
    for i in range(M + 1):
        inf_rate[i] = get_inf_rate(mode = theta_pre.mean(dim = 0)[i].item(), std_orig = theta_pre.std(dim = 0)[i].item(),
                        lower = lower[i].item(), upper = upper[i].item(), actual_inf_rate = actual_inf_rate[i].item())

    mean_theta = theta_pre.mean(dim = 0)
    std_theta = inf_rate * theta_pre.std(dim = 0)

    # ============= Main Training Data ============== #
    sample_size = training_size 
    ### Read the same training data
    theta_r0 = np.load(f'single_training_data/theta_r0_task{task_id}.npy', mmap_mode='r')[:sample_size]
    data_r0  = np.load(f'single_training_data/data_r0_task{task_id}.npy',  mmap_mode='r')[:sample_size]
    theta_r0 = torch.tensor(theta_r0, dtype = torch.float32).contiguous()
    data_r0 = torch.tensor(data_r0, dtype = torch.float32).contiguous()

    # Generate the validation data
    val_theta_r0, val_data_r0 = gen_ref_distinct_theta(mean_theta, std_theta, lower, upper, int(1e6) + 100)

    bad_mask = torch.isinf(val_theta_r0).any(dim=1)
    val_theta_r0 = val_theta_r0[~bad_mask]
    val_data_r0 = val_data_r0[~bad_mask]

    val_theta_r0 = val_theta_r0[:int(1e6)]
    val_data_r0 = val_data_r0[:int(1e6)]

    print('traing size:', theta_r0.shape[0])
    print('validation size:', val_theta_r0.shape[0])

    # ============= Extra Data ============== #
    theta_r0_extra = torch.from_numpy(np.load(f'extra_data/theta_r0_extra_task{task_id}.npy'))[:extra_sample_size]
    data_r0_extra = torch.from_numpy(np.load(f'extra_data/data_r0_extra_task{task_id}.npy'))[:extra_sample_size]

    val_theta_r0_extra, val_data_r0_extra = gen_ref(mean_theta, std_theta, lower, upper, extra_obs_size, extra_sample_size + 100)

    bad_mask = torch.isinf(val_theta_r0_extra).any(dim=1)
    val_theta_r0_extra = val_theta_r0_extra[~bad_mask]
    val_data_r0_extra = val_data_r0_extra[~bad_mask]
    val_theta_r0_extra = val_theta_r0_extra[:extra_sample_size]
    val_data_r0_extra = val_data_r0_extra[:extra_sample_size]

    print('extra size:', theta_r0_extra.shape[0])
    print('val extra size:', val_theta_r0_extra.shape[0])


    # ============= Proposal Score ============== #
    prop_score_r0 = ( mean_theta.repeat(theta_r0.shape[0], 1) - theta_r0 ) / (std_theta**2).repeat(theta_r0.shape[0], 1)
    val_prop_score_r0 = ( mean_theta.repeat(val_theta_r0.shape[0], 1) - val_theta_r0 ) / (std_theta**2).repeat(val_theta_r0.shape[0], 1)

    #####################################################
    #          Determine the weight function            #
    #####################################################

    scale = dist2bd(theta_r0.to(device), data_r0.to(device))[0].mean(dim = 0, keepdim = True)

    def make_g_functions(scale):
        def g(theta, x):
            d, lower, upper = dist2bd(theta, x)
            return d / scale

        def g1(theta, x):
            d, lower, upper = dist2bd(theta, x)
            return (2 * (theta < (lower + upper) / 2) - 1) / scale # 1 or -1, depends on closer to the lower or upper bound
        
        return g, g1

    g, g1 = make_g_functions(scale)
        


    #######################################
    #          Training starts            #
    #######################################

    ### Train the NN
    # Create DataLoader

    dataset = TensorDataset(theta_r0, data_r0, prop_score_r0)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    val_dataset = TensorDataset(val_theta_r0, val_data_r0, val_prop_score_r0)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    dataset_extra = TensorDataset(theta_r0_extra, data_r0_extra)
    dataloader_extra = DataLoader(dataset_extra, batch_size = batch_size_extra, shuffle = True)
    val_dataset_extra = TensorDataset(val_theta_r0_extra, val_data_r0_extra)
    val_dataloader_extra = DataLoader(val_dataset_extra, batch_size = batch_size_extra, shuffle = False)

    # Create model and optimizer
    theta_dim = M + 1 
    x_dim = 2 
    obs_size = data_obs.shape[0] 
    print(f"\n hidden_size = {hidden_size}, num_layers = {num_layers}, learning_rate = {learning_rate} \n")

    # continue training from the initialized model
    model = single_ELU_LikeScoreMatchingNN(theta_dim, x_dim, obs_size, hidden_size, num_layers)
    checkpoint = torch.load(f'model_single_init/checkpoint_task{task_id}_trainsize{training_size}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # CHECK LOSS
    print("Loss of the initial model")
    check_loss(model, val_dataloader, val_dataloader_extra, lam_fisher)
    print("\n")


    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

    scheduler = None

    bias_lastlayer = train_fisher(model, optimizer, dataloader, val_dataloader, dataloader_extra, val_dataloader_extra, lam_fisher, g, g1, num_epochs, scheduler)


    os.makedirs('model_single_fisher', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bias_lastlayer': bias_lastlayer
    }, f'model_single_fisher/checkpoint_task{task_id}_trainsize{training_size}.pth')


    #############################################
    #          Record the total time            #
    #############################################
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total time: {round(total_duration/3600, 2)} hours')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)
