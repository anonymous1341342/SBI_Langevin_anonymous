import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import copy
from pathlib import Path
from itertools import cycle
from pathlib import Path
from torch.autograd.functional import jacobian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a1 = 0.0
b1 = 10.0

a2 = 0.0
b2 = 10.0

a3 = 0.01 # 0.0
b3 = 0.5

# Generate the reference table
def gen_ref_table(a1, a2, a3, b1, b2, b3, dim = 5, obs_size = 500, sample_size = 10000):
    # a, b: prior of theta
    
    # generate the parameters
    theta1 = np.random.uniform(low = a1, high = b1, size = sample_size)
    theta2 = np.random.uniform(low = a2, high = b2, size = sample_size)
    theta3 = np.random.uniform(low = a3, high = b3, size = sample_size)

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

    theta = np.c_[theta1, theta2, theta3]
    return torch.tensor(theta, dtype = torch.float32), torch.tensor(x_stretched, dtype = torch.float32)




# Given the true parameters (theta1, theta2, theta3), generate observed data of size 'obs_size'
def gen_obs_data(theta1, theta2, theta3, dim = 5, obs_size = 500): # generate observed data under fixed (true) parameters
    # dim: dimention of each observed data point
    # obs_size: number of observed samples
    # theta1, theta2, theta3: scalars
    
    # generate w and u
    w = np.random.exponential(scale = 1.0/theta3, size = (obs_size, dim)) # scale = inverse rate
    u = np.random.uniform(low = theta1, high = theta1 + theta2, size = (obs_size, dim)) 

    # use w and u to calculate x
    x = np.zeros((obs_size, dim))
    x[:, 0] = u[:, 0] + w[:, 0]
    for k in range(1, dim):
        # tmp = np.sum(w[:, :(k+1)], axis = 1) - np.sum(x[:, :k], axis = 1) # take k+1, as right boundary is not included
        x[:, k] = u[:, k] + np.maximum(0, np.sum(w[:, :(k+1)], axis = 1) - np.sum(x[:, :k], axis = 1))
    return torch.tensor(x, dtype = torch.float32)



# The neural network, Tanh() activation
class Tanh_nmodel_LikeScoreMatchingNN(nn.Module):
    def __init__(self, theta_dim, x_dim, obs_size, hidden_size, num_layers):
        super(Tanh_nmodel_LikeScoreMatchingNN, self).__init__()

        layers = [nn.Linear(theta_dim + x_dim, hidden_size), nn.Tanh()]
        
        # Add hidden layers based on num_layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer to match the desired output size
        layers.append(nn.Linear(hidden_size, theta_dim))

        self.layers = nn.Sequential(*layers)
        self.x_dim = x_dim
        self.theta_dim = theta_dim
        self.obs_size = obs_size

    def forward(self, theta, x):
        if len(theta.shape) == 1: # if one-dimensional
            theta = theta.view(-1, 1)

        if len(x.shape) == 1:
            x = x.view(-1, 1)
        x_dim = self.x_dim
        score = self.layers( torch.cat( (theta.repeat_interleave(self.obs_size, dim = 0), x.reshape(-1, x_dim)), dim = 1 ) ).view(theta.shape[0], self.obs_size, theta.shape[1]).sum(dim = 1)
        return score
    



# we will add a weight function g() to meet the boundary condition
def Like_score_loss(model, theta, x, prop_score, g, g1):
    # g: weight function, takes input (theta, x), output dimension is the same as theta
    # g1: first derivative of g (the diagonal part, \partial g(\theta, x)_j / \partial \theta_j), output dimension is the same as theta
    # we require g and g1 to be able to address matrix input and do element-wise mapping

    score = model(theta, x)
    loss1 = (score * g(theta, x)**(1/2)) ** 2 / 2. # [B, theta_dim]
    loss3 = ((score * g(theta, x)) * prop_score) # [B, theta_dim]
    
    theta.requires_grad_(True)
    score_tmp = model(theta, x) # In order to calculate grad2
    loss2 = torch.zeros_like(theta)
    for i in range(theta.shape[1]):
        grad2 = torch.autograd.grad(outputs = score_tmp[:, i].sum(), inputs = theta, create_graph=True)[0][:, i]
        loss2[:, i] = grad2 * (g(theta, x)[:, i]) + score[:, i] * g1(theta, x)[:, i]
    
    loss = loss1 + loss2 + loss3 # [B, theta_dim]
    # the first one is the score matching loss, the third one is the score matching loss on each dimension 
    return loss.mean(dim = 0).sum() / model.obs_size, loss.mean(dim = 0) / model.obs_size 


def train(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience, return_best_model = False):
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None

    # record training loss and validation loss at each epoch and then plot
    start_time = time.time()
    path_loss_all_dim = []
    path_val_loss_all_dim = []
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train() 
        total_loss = 0.0
        valid_batches = 0
        total_loss_alldim = torch.zeros(model.theta_dim).to(device)
        for batch_sample in dataloader:
            optimizer.zero_grad()
            batch_theta, batch_x, batch_prop_score = batch_sample
            batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
            loss, loss_alldim = Like_score_loss(model, batch_theta, batch_x, batch_prop_score, g, g1)
            if torch.isnan(loss):
                print(f"[WARNING] NaN detected, skipping this minibatch")
                continue
            valid_batches += 1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_alldim += loss_alldim.detach() 
        avg_loss = total_loss / valid_batches
        avg_total_loss_alldim = total_loss_alldim / valid_batches
        path_loss_all_dim.append(avg_total_loss_alldim.cpu().numpy())

        model.eval()
        total_loss_val = 0.0
        total_loss_alldim_val = torch.zeros(model.theta_dim).to(device)
        val_valid_batches = 0
        for val_batch_sample in val_dataloader:
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_sample
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_theta.to(device), val_batch_x.to(device), val_batch_prop_score.to(device)
            val_loss, val_loss_alldim = Like_score_loss(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
            if torch.isnan(val_loss):
                # print(f"[WARNING] NaN detected, skipping this minibatch")
                continue
            val_valid_batches += 1
            total_loss_val += val_loss.item()    
            total_loss_alldim_val += val_loss_alldim.detach()

        avg_val_loss = total_loss_val / val_valid_batches
        avg_total_loss_alldim_val = total_loss_alldim_val / val_valid_batches
        path_val_loss_all_dim.append(avg_total_loss_alldim_val.cpu().numpy())
        if avg_val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = avg_val_loss
            best_val_loss_alldim = avg_total_loss_alldim_val
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        

        time2 = time.time()
        if epoch % 1 == 0 or epoch == num_epochs:
            print(f'Epoch {epoch+1}/{num_epochs} | Training Loss: {round(avg_loss, 3)} | Validation Loss: {round(avg_val_loss, 3)} | Time: {round(time2 - time1, 2)} seconds')
            print(f'Training Loss (alldim): {np.round(avg_total_loss_alldim.cpu().numpy(), 4)}\nValidation Loss (alldim): {np.round(avg_total_loss_alldim_val.cpu().numpy(), 4)}\n')
        
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"Epoch {epoch+1}: reducing learning rate to {new_lr:.2e}")
        
        
        # early stop
        if (epoch+1) - best_epoch >= early_stop_patience:
            print(f"Val_loss didn't improve after {early_stop_patience} epochs, stop training")
            break

    # Load best model state after training
    if return_best_model and best_model_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        print(f"Return the best model at epoch {best_epoch}, with Validation Loss: {best_val_loss:.3f}")


    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total training time: {round(total_duration, 2)} seconds')
    return path_val_loss_all_dim, path_loss_all_dim



## The weight function
# returns the distance of \theta_j to the boundary when fixing (\theta_{-j}, x)
def dist2bd(theta, x):
    # input of theta and x can have n rows, each row is an observation
    
    # we first calculate the boundary, including lower and upper bound
    # lower and upper bounds for theta1
    lower1 = a1 * torch.ones(theta.shape[0]).view(-1, 1).to(device)
    upper1 = torch.min(x.min(dim = 1).values.view(-1, 1), b1 * torch.ones(theta.shape[0]).view(-1, 1).to(device))

    # for theta2
    lower2 = a2 * torch.ones(theta.shape[0]).view(-1, 1).to(device)
    upper2 = b2 * torch.ones(theta.shape[0]).view(-1, 1).to(device)

    # for theta3
    lower3 = a3 * torch.ones(theta.shape[0]).view(-1, 1).to(device)
    upper3 = b3 * torch.ones(theta.shape[0]).view(-1, 1).to(device)

    lower = torch.cat((lower1, lower2, lower3), dim=1)
    upper = torch.cat((upper1, upper2, upper3), dim=1)
    
    # return the distance to the boundary, and also the lower and upper bound
    return torch.min(theta - lower, upper - theta), lower, upper

# h(t) = t, identity map
def g(theta, x):
    d, lower, upper = dist2bd(theta, x)
    return d

def g1(theta, x):
    d, lower, upper = dist2bd(theta, x)
    return (2 * (theta < (lower + upper) / 2) - 1) # 1 or -1, depends on closer to the lower or upper bound



