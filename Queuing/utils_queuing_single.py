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
from itertools import cycle
from torch.autograd.functional import jacobian
import copy
from itertools import cycle
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a1 = 0.0
b1 = 10.0

a2 = 0.0
b2 = 10.0

a3 = 0.01 # 
b3 = 0.5

# Generate the ref_R table for mean-regression and curvature penalty, each theta generates n observed data points
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
            x[:, k] = u[:, k] + np.maximum(0, np.sum(w[:, :(k+1)], axis = 1) - np.sum(x[:, :k], axis = 1))
        x_stretched[:, (j * dim):((j+1) * dim)] = x # stack all the observed x

    theta = np.c_[theta1, theta2, theta3]
    return torch.tensor(theta, dtype = torch.float32), torch.tensor(x_stretched, dtype = torch.float32)

# Generate the ref_S table for score matching, each theta generates one observed x
def gen_ref_table_distinct_theta(a1, a2, a3, b1, b2, b3, dim = 5, sample_size = 10000):
    # a, b: prior of theta
    
    # generate the parameters
    theta1 = np.random.uniform(low = a1, high = b1, size = sample_size)
    theta2 = np.random.uniform(low = a2, high = b2, size = sample_size)
    theta3 = np.random.uniform(low = a3, high = b3, size = sample_size)

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
        x[:, k] = u[:, k] + np.maximum(0, np.sum(w[:, :(k+1)], axis = 1) - np.sum(x[:, :k], axis = 1))

    theta = np.c_[theta1, theta2, theta3]
    return torch.tensor(theta, dtype = torch.float32), torch.tensor(x, dtype = torch.float32)

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
        x[:, k] = u[:, k] + np.maximum(0, np.sum(w[:, :(k+1)], axis = 1) - np.sum(x[:, :k], axis = 1))
    return torch.tensor(x, dtype = torch.float32)



## The weight function
def dist2bd(theta, x):
    # input of theta and x can have n rows, each row is an observation
    
    # we first calculate the boundary, including lower and upper bound
    # lower and upper bounds for theta1
    device = theta.device
    lower1 = a1 * torch.ones(theta.shape[0]).view(-1, 1).to(device)
    # We will add a gaussian noise to x, so the domain of x does not depend on theta
    upper1 = b1 * torch.ones(theta.shape[0]).view(-1, 1).to(device)

    # for theta2
    lower2 = a2 * torch.ones(theta.shape[0]).view(-1, 1).to(device)
    upper2 = b2 * torch.ones(theta.shape[0]).view(-1, 1).to(device)

    # for theta3
    lower3 = a3 * torch.ones(theta.shape[0]).view(-1, 1).to(device)
    upper3 = b3 * torch.ones(theta.shape[0]).view(-1, 1).to(device)

    lower = torch.cat((lower1, lower2, lower3), dim=1)
    upper = torch.cat((upper1, upper2, upper3), dim=1)
    
    return torch.min(theta - lower, upper - theta), lower, upper

def g(theta, x):
    d, lower, upper = dist2bd(theta, x)
    return d

def g1(theta, x):
    d, lower, upper = dist2bd(theta, x)
    return (2 * (theta < (lower + upper) / 2) - 1) # 1 or -1, depends on closer to the lower or upper bound



# The neural network, ELU() activation
class ELU_single_LikeScoreMatchingNN(nn.Module):
    def __init__(self, theta_dim, x_dim, hidden_size, num_layers):
        super(ELU_single_LikeScoreMatchingNN, self).__init__()

        layers = [nn.Linear(theta_dim + x_dim, hidden_size), nn.ELU()]
        
        # Add hidden layers based on num_layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ELU())

        # Output layer to match the desired output size
        layers.append(nn.Linear(hidden_size, theta_dim))

        self.layers = nn.Sequential(*layers)
        self.x_dim = x_dim
        self.theta_dim = theta_dim

    def forward(self, theta, x):
        if len(theta.shape) == 1: # if one-dimensional
            theta = theta.view(-1, 1)
        if len(x.shape) == 1:
            x = x.view(-1, 1)
        
        return self.layers(torch.cat((theta, x), dim = 1))

    def cal_penalty(self, theta, x):
        if len(theta.shape) == 1: # if one-dimensional
            theta = theta.view(-1, 1)

        if len(x.shape) == 1:
            x = x.view(-1, 1)
        x_dim = self.x_dim
        obs_size = int(x.shape[1] / x_dim)
        score_tensor = self.layers( torch.cat( (theta.repeat_interleave(obs_size, dim = 0), x.reshape(-1, x_dim)), dim = 1 ) ).view(theta.shape[0], obs_size, theta.shape[1])
        return score_tensor




def Like_score_loss_deb(model, theta, x, prop_score, g, g1):
    # g: weight function, takes input (theta, x), output dimension is the same as theta
    # g1: first derivative of g (the diagonal part, \partial g(\theta, x)_j / \partial \theta_j), output dimension is the same as theta
    # we require g and g1 to be able to address matrix input and do element-wise mapping

    # bias = model(theta, x).mean(dim = 0)
    # score = model(theta, x) - bias
    score = model(theta, x)
    bias = score.mean(dim = 0)
    score = score - bias


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
    return loss.mean(dim = 0).sum(), bias, loss.mean(dim = 0)



def train_deb(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience, return_best_model = False):
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
            loss, bias, loss_alldim = Like_score_loss_deb(model, batch_theta, batch_x, batch_prop_score, g, g1)
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
            val_loss, val_bias, val_loss_alldim = Like_score_loss_deb(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
            if torch.isnan(val_loss):
                print(f"[WARNING] NaN detected, skipping this minibatch")
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
    print(f'Total training time: {round(total_duration, 2)} seconds')
    return bias_lastlayer, path_val_loss_all_dim, path_loss_all_dim




# Use the variance of score
def weighted_Fisher_penalty(model, theta_extra, x_extra, g):
    # g: weight function, takes input (theta, x), output dimension is the same as theta

    def score_sum_fn(theta_): # in order to calculate Jacobian
        return ( model.cal_penalty(theta_, x_extra).sum(dim = 1) ).sum(dim = 0)
    
    theta_extra.requires_grad_(True)
    obs_size = int(x_extra.shape[1] / model.x_dim) # extra_obs_size
    E_Jacobian = jacobian(score_sum_fn, theta_extra, create_graph = True).permute(1, 0, 2) / obs_size # estimated mean of the Jacobian, of dimension (batch_size, theta_dim, theta_dim)

    score_tensor_extra = model.cal_penalty(theta_extra, x_extra)
    
    EssT = torch.einsum('ibc,ibd->icd', score_tensor_extra - score_tensor_extra.mean(dim = 1, keepdim = True), score_tensor_extra - score_tensor_extra.mean(dim = 1, keepdim = True)) / obs_size 

    weight_mat = g(theta_extra, x_extra)**(1/2) # the same dimension as theta_extra, (b, d)
    weight_tensor = weight_mat.unsqueeze(2) @ weight_mat.unsqueeze(1) # (b, d, d)
    penalty_fisher = ( (EssT * weight_tensor + E_Jacobian * weight_tensor)**2 ).sum(dim = (1, 2)) # a vector storing the Fnorm^2 for every row in the minibatch
    
    return penalty_fisher.mean()



# use the variance of score
def cal_weighted_ssT(model, theta_extra, x_extra, g):
    # calculate E[||E(ssT)||_F^2]

    score_tensor_extra = model.cal_penalty(theta_extra, x_extra)
    obs_size = int(x_extra.shape[1] / model.x_dim) # extra_obs_size
    
    EssT = torch.einsum('ibc,ibd->icd', score_tensor_extra - score_tensor_extra.mean(dim = 1, keepdim = True), score_tensor_extra - score_tensor_extra.mean(dim = 1, keepdim = True)) / obs_size 

    weight_mat = g(theta_extra, x_extra)**(1/2) # the same dimension as theta_extra, (b, d)
    weight_tensor = weight_mat.unsqueeze(2) @ weight_mat.unsqueeze(1)
    
    return ( (EssT * weight_tensor)**2 ).sum(dim = (1, 2)).mean()


# =============== Debias Regression Model and Training Functions =============== #
class Deb_ELU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers):
        super().__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ELU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ELU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



# score * g(theta, x)**(1/2)
def train_weighted_DebReg(model, optimizer, train_loader, val_loader, num_epochs, scheduler, early_stop_patience):
    model.to(device)
    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None
    best_epoch = None

    start_time = time.time()
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        total_train_loss = 0.0

        for xb, yb, weight in train_loader:
            xb, yb, weight = xb.to(device), yb.to(device), weight.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred * weight**(1/2), yb * weight**(1/2))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, weight in val_loader:
                xb, yb, weight = xb.to(device), yb.to(device), weight.to(device)
                pred = model(xb)
                loss = loss_fn(pred * weight**(1/2), yb * weight**(1/2))
                total_val_loss += loss.item() * xb.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # Save best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())  # Save a copy of the model weights
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch + 1

        # scheduler step
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                print(f"Epoch {epoch+1}: reducing learning rate to {new_lr:.2e}")
        
        time2 = time.time()
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {round(time2 - time1, 2)} seconds")

        # early stop
        if (epoch+1) - best_epoch >= early_stop_patience:
            print(f"Val_loss didn't improve after {early_stop_patience} epochs, stop training")
            break

    print(f"Return the model at epoch {best_epoch}, with validation loss {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    end_time = time.time()
    print(f"Total training time of the Deb Reg model = {(end_time - start_time)/60:.2f} minutes")


def Deb_curve(model, x):
    # get the jacobian of the DebReg model

    def reg_output(x_): # in order to calculate Jacobian
        return ( model(x_) ).sum(dim = 0)
    
    x.requires_grad_(True)
    # jacob is for the minibatch data, has dim (b, d, d)
    jacob = jacobian(reg_output, x, create_graph = True).permute(1, 0, 2)
    
    return jacob



def train_weighted_DebReg_fisher_crossterm(model, optimizer, train_loader, val_loader, num_epochs, lam_curve, best_val_reg_loss):
    model.to(device)
    loss_fn = nn.MSELoss()

    # best_val_reg_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None
    best_epoch = None

    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())

    start_time = time.time()
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        total_obj = 0.0
        total_reg_loss = 0.0
        total_curve_loss = 0.0

        for xb, yb, weight in train_loader:
            xb, yb, weight = xb.to(device), yb.to(device), weight.to(device)
            weight = weight**(1/2) # g**(1/2)
            weight_tensor = weight.unsqueeze(2) @ weight.unsqueeze(1) # (b, d, d)
            optimizer.zero_grad()
            
            pred = model(xb)
            pred_predT = pred.unsqueeze(2) @ pred.unsqueeze(1)
            jacob = Deb_curve(model, xb)
            loss_reg = loss_fn(pred * weight, yb * weight)
            loss_curve = loss_fn( pred_predT * weight_tensor, (jacob + torch.einsum('bi,bj->bij', yb, pred) + torch.einsum('bi,bj->bij', pred, yb)) * weight_tensor )
            loss = loss_reg + lam_curve * loss_curve
            
            loss.backward()
            optimizer.step()
            total_obj += loss.item() * xb.size(0)
            total_reg_loss += loss_reg.item() * xb.size(0)
            total_curve_loss += loss_curve.item() * xb.size(0)
            
        avg_obj = total_obj / len(train_loader.dataset)
        avg_reg_loss = total_reg_loss / len(train_loader.dataset)
        avg_curve_loss = total_curve_loss / len(train_loader.dataset)

        
        # Validation
        model.eval()
        val_total_obj = 0.0
        val_total_reg_loss = 0.0
        val_total_curve_loss = 0.0
        # with torch.no_grad():
        for xb, yb, weight in val_loader:
            xb, yb, weight = xb.to(device), yb.to(device), weight.to(device)
            weight = weight**(1/2) # g**(1/2)
            weight_tensor = weight.unsqueeze(2) @ weight.unsqueeze(1) # (b, d, d)
            
            pred = model(xb)
            pred_predT = pred.unsqueeze(2) @ pred.unsqueeze(1)
            jacob = Deb_curve(model, xb)
            loss_reg = loss_fn(pred * weight, yb * weight)
            loss_curve = loss_fn( pred_predT * weight_tensor, (jacob + torch.einsum('bi,bj->bij', yb, pred) + torch.einsum('bi,bj->bij', pred, yb)) * weight_tensor )
            loss = loss_reg + lam_curve * loss_curve

            val_total_obj += loss.item() * xb.size(0)
            val_total_reg_loss += loss_reg.item() * xb.size(0)
            val_total_curve_loss += loss_curve.item() * xb.size(0)

        avg_val_obj = val_total_obj / len(val_loader.dataset)
        avg_val_reg_loss = val_total_reg_loss / len(val_loader.dataset)
        avg_val_curve_loss = val_total_curve_loss / len(val_loader.dataset)

        # Save best model (in terms of val_reg_loss) so far
        if avg_val_reg_loss < best_val_reg_loss:
            best_val_reg_loss = avg_val_reg_loss
            best_model_state = copy.deepcopy(model.state_dict())  # Save a copy of the model weights
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch + 1
        
        time2 = time.time()
        print(f"Epoch {epoch+1} | Train Loss (Total, Reg, Curve): ({avg_obj:.6f}, {avg_reg_loss:.6f},{avg_curve_loss:.6f}) | Val Loss (Total, Reg, Curve): ({avg_val_obj:.6f}, {avg_val_reg_loss:.6f},{avg_val_curve_loss:.6f}) | Time: {round(time2 - time1, 2)} seconds")


    if best_model_state is not None:
        print(f"Returning the best model at epoch {best_epoch}, with validation Reg loss {best_val_reg_loss:.4f}")
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
    else:
        print(f"No improvement over best_val_reg_loss = {best_val_reg_loss:.4f}. Reverting to initial model.")
        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optimizer_state)

    end_time = time.time()
    print(f"Total training time of the DebReg_curve model = {(end_time - start_time)/60:.2f} minutes")










