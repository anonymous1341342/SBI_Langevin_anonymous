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
import math
import pandas as pd
from scipy.stats import truncnorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ELU_single_LikeScoreMatchingNN_sparse(nn.Module):
    """
    Parallel implementation of separate MLPs using grouped Conv1d — no for loop.
    Equivalent to theta_dim independent MLPs, but all computed in parallel.
    """
    def __init__(self, theta_dim, x_dim, hidden_size, num_layers):
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        in_dim = theta_dim + x_dim

        # grouped Conv1d with groups=theta_dim simulates independent MLPs:
        # Conv1d(in, out, kernel=1, groups=g) is equivalent to g independent Linear(in/g, out/g)
        # so we tile the input theta_dim times to give each head its own channels

        # First layer: in_dim -> hidden_size per head
        layers = []
        layers.append(nn.Conv1d(theta_dim * in_dim, theta_dim * hidden_size,
                                kernel_size=1, groups=theta_dim))
        layers.append(nn.ELU())

        # Intermediate layers: hidden_size -> hidden_size per head
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(theta_dim * hidden_size, theta_dim * hidden_size,
                                    kernel_size=1, groups=theta_dim))
            layers.append(nn.ELU())

        # Final layer: hidden_size -> 1 per head
        layers.append(nn.Conv1d(theta_dim * hidden_size, theta_dim,
                                kernel_size=1, groups=theta_dim))

        self.net = nn.Sequential(*layers)
        self._in_dim = in_dim

    def forward(self, theta, x):
        if theta.ndim == 1:
            theta = theta.view(1, -1)
        if x.ndim == 1:
            x = x.view(1, -1)

        B = theta.shape[0]
        inp = torch.cat((theta, x), dim=1)  # (B, in_dim)

        # Each head receives the full input — tile theta_dim times
        # shape: (B, theta_dim * in_dim, 1)
        inp_expanded = inp.unsqueeze(1).expand(B, self.theta_dim, self._in_dim)
        inp_expanded = inp_expanded.reshape(B, self.theta_dim * self._in_dim, 1)

        out = self.net(inp_expanded)  # (B, theta_dim, 1)
        return out.squeeze(-1)        # (B, theta_dim)

    def cal_penalty(self, theta, x):
        if theta.ndim == 1:
            theta = theta.view(1, -1)
        if x.ndim == 1:
            x = x.view(1, -1)

        x_dim = self.x_dim
        obs_size = int(x.shape[1] / x_dim)
        batch_size = theta.shape[0]

        theta_repeated = theta.repeat_interleave(obs_size, dim=0)  # (batch*obs_size, theta_dim)
        x_flat = x.reshape(-1, x_dim)                              # (batch*obs_size, x_dim)

        out = self.forward(theta_repeated, x_flat)                 # (batch*obs_size, theta_dim)

        return out.view(batch_size, obs_size, self.theta_dim)      # (batch, obs_size, theta_dim)


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

        model.eval()
        total_loss_val = 0.0
        total_loss_alldim_val = torch.zeros(model.theta_dim).to(device)
        val_valid_batches = 0
        for val_batch_sample in val_dataloader:
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_sample
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_theta.to(device), val_batch_x.to(device), val_batch_prop_score.to(device)
            val_loss, val_bias, val_loss_alldim = Like_score_loss_deb(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
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
    print(f'Total training time: {round(total_duration, 2)} seconds')
    return bias_lastlayer, path_val_loss_all_dim





# stop update the networks for dims that already converged
def Like_score_loss_deb_freeze(model, theta, x, prop_score, g, g1, active_dims):
    # g: weight function, takes input (theta, x), output dimension is the same as theta
    # g1: first derivative of g (the diagonal part, \partial g(\theta, x)_j / \partial \theta_j), output dimension is the same as theta
    # we require g and g1 to be able to address matrix input and do element-wise mapping
    # active_dims is a boolean tensor

    score = model(theta, x)
    bias = score.mean(dim = 0)
    score = score - bias


    loss1 = (score * g(theta, x)**(1/2)) ** 2 / 2. # [B, theta_dim]
    loss3 = ((score * g(theta, x)) * prop_score) # [B, theta_dim]
    
    theta.requires_grad_(True)
    score_tmp = model(theta, x) # In order to calculate grad2
    loss2 = torch.zeros_like(theta)
    for i in torch.where(active_dims)[0]:
        grad2 = torch.autograd.grad(outputs = score_tmp[:, i].sum(), inputs = theta, create_graph=True)[0][:, i]
        loss2[:, i] = grad2 * (g(theta, x)[:, i]) + score[:, i] * g1(theta, x)[:, i]
    
    loss = loss1 + loss2 + loss3 # [B, theta_dim]
    # the first one is the score matching loss, the third one is the score matching loss on each dimension 
    return loss.mean(dim = 0)[active_dims].sum(), bias, loss.mean(dim = 0)


def train_deb_freeze(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience):
    model.to(device)
    active_dims = torch.ones(model.theta_dim, dtype=torch.bool).to(device) # all dimensions are active at the beginning
    frozen_val_loss_all_dim = torch.zeros(model.theta_dim).to(device) # to store the loss of the frozen dimensions
    frozen_loss_all_dim = torch.zeros(model.theta_dim).to(device) # to store the loss of the frozen dimensions
    best_val_loss_alldim = float('inf') * torch.ones(model.theta_dim).to(device) # for all dim
    best_epoch = torch.zeros(model.theta_dim, dtype=torch.long).to(device) # for all dim

    # To store the weights of the MLPs of the dims that are freezed
    # key: dim_index, value: {param_name: tensor}
    frozen_param_snapshots = {}

    def get_dim_slice_map():
        """
        A mapping of dim_index -> [(param, slice)], only for the sparse Conv1d nn
        """
        if not hasattr(model, 'net'):
            return None
        dim_map = {i: [] for i in range(model.theta_dim)}
        for layer in model.net:
            if isinstance(layer, nn.Conv1d):
                out_per_group = layer.out_channels // model.theta_dim
                for i in range(model.theta_dim):
                    sl = slice(i * out_per_group, (i + 1) * out_per_group)
                    dim_map[i].append((layer.weight, sl))
                    if layer.bias is not None:
                        dim_map[i].append((layer.bias, sl))
        return dim_map

    dim_slice_map = get_dim_slice_map()


    # record training loss and validation loss at each epoch and then plot
    start_time = time.time()
    path_val_loss_all_dim = []
    path_loss_all_dim = []
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
            loss, bias, loss_alldim = Like_score_loss_deb_freeze(model, batch_theta, batch_x, batch_prop_score, g, g1, active_dims)
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

        model.eval()
        total_loss_val = 0.0
        total_loss_alldim_val = torch.zeros(model.theta_dim).to(device)
        val_valid_batches = 0
        for val_batch_sample in val_dataloader:
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_sample
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_theta.to(device), val_batch_x.to(device), val_batch_prop_score.to(device)
            val_loss, val_bias, val_loss_alldim = Like_score_loss_deb_freeze(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1, active_dims)
            if torch.isnan(val_loss):
                # print(f"[WARNING] NaN detected, skipping this minibatch")
                continue
            val_valid_batches += 1
            total_loss_val += val_loss.item()    
            total_loss_alldim_val += val_loss_alldim.detach()

        avg_val_loss = total_loss_val / val_valid_batches
        avg_total_loss_alldim_val = total_loss_alldim_val / val_valid_batches
        # path_val_loss_all_dim.append(avg_total_loss_alldim_val.cpu().numpy()) 

        # 1. Update best_epoch and best_val_loss
        best_epoch[avg_total_loss_alldim_val < best_val_loss_alldim] = epoch
        best_val_loss_alldim = torch.minimum(best_val_loss_alldim, avg_total_loss_alldim_val)

        # 2. Check which dimensions should be frozen
        should_freeze = (epoch - best_epoch >= early_stop_patience) & active_dims

        # store the weights for the frozen dimensions
        if dim_slice_map is not None:
            for i in torch.where(should_freeze)[0].tolist():
                frozen_param_snapshots[i] = [
                    (param, sl, param.data[sl].clone())
                    for param, sl in dim_slice_map[i]
                ]


        # 3. Record frozen_loss 
        frozen_loss_all_dim[should_freeze] = avg_total_loss_alldim[should_freeze]
        frozen_val_loss_all_dim[should_freeze] = avg_total_loss_alldim_val[should_freeze]

        # 4. freeze
        active_dims[should_freeze] = False

        # 5. Replace the loss of inactive dimensions with their frozen loss for display
        avg_total_loss_alldim[~active_dims] = frozen_loss_all_dim[~active_dims]
        avg_total_loss_alldim_val[~active_dims] = frozen_val_loss_all_dim[~active_dims]

        path_val_loss_all_dim.append(avg_total_loss_alldim_val.cpu().numpy()) 
        path_loss_all_dim.append(avg_total_loss_alldim.cpu().numpy())

        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"Epoch {epoch+1}: reducing learning rate to {new_lr:.2e}")
        
        time2 = time.time()
        if epoch % 1 == 0 or epoch == num_epochs:
            print(f'Epoch {epoch+1}/{num_epochs} | Training Loss: {round(avg_loss+frozen_loss_all_dim.sum().item(), 3)} | Validation Loss: {round(avg_val_loss+frozen_val_loss_all_dim.sum().item(), 3)} | Time: {round(time2 - time1, 2)} seconds')
            print(f'Training Loss (alldim): {np.round(avg_total_loss_alldim.cpu().numpy(), 4)}\nValidation Loss (alldim): {np.round(avg_total_loss_alldim_val.cpu().numpy(), 4)}\n')

        if active_dims.sum() == 0:
            print("All dimensions have converged, stopping training")
            break
    
    # After training, load the saved weights for the frozen dims
    with torch.no_grad():
        for i, snapshots in frozen_param_snapshots.items():
            for param, sl, saved_val in snapshots:
                param.data[sl] = saved_val
    if frozen_param_snapshots:
        print(f"Restored parameters for frozen dims: {sorted(frozen_param_snapshots.keys())}")



    # output the final model, we just need to minus the bias
    # we calculate the bias using the whole dataset
    active_dims = torch.ones(model.theta_dim, dtype=torch.bool).to(device)
    total_bias = 0.0 # is actually a vector of the same dimension as theta
    for batch_sample in dataloader:
        batch_theta, batch_x, batch_prop_score = batch_sample
        batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
        loss, bias, _ = Like_score_loss_deb_freeze(model, batch_theta, batch_x, batch_prop_score, g, g1, active_dims)
        total_bias += bias.detach()
    # with torch.no_grad(): 
    #     model.layers[-1].bias -= (total_bias / len(dataloader)).to(device) 

    bias_lastlayer = total_bias / len(dataloader)
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total training time: {round(total_duration, 2)} seconds')
    return bias_lastlayer, path_val_loss_all_dim, path_loss_all_dim






def weighted_Fisher_penalty(model, theta_extra, x_extra, g):
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



def cal_weighted_ssT(model, theta_extra, x_extra, g):
    # calculate E[||E(ssT)||_F^2]

    score_tensor_extra = model.cal_penalty(theta_extra, x_extra)
    obs_size = int(x_extra.shape[1] / model.x_dim) # extra_obs_size
    
    EssT = torch.einsum('ibc,ibd->icd', score_tensor_extra - score_tensor_extra.mean(dim = 1, keepdim = True), score_tensor_extra - score_tensor_extra.mean(dim = 1, keepdim = True)) / obs_size 

    weight_mat = g(theta_extra, x_extra)**(1/2) # the same dimension as theta_extra, (b, d)
    weight_tensor = weight_mat.unsqueeze(2) @ weight_mat.unsqueeze(1)
    
    return ( (EssT * weight_tensor)**2 ).sum(dim = (1, 2)).mean()


# =============== Debias Regression Model and Training Functions =============== #
class Deb_ELU_sparse(nn.Module):
    """
    Parallel implementation of separate MLPs using grouped Conv1d — no for loop.
    Equivalent to output_dim independent MLPs, each mapping input_dim -> 1.
    """
    def __init__(self, input_dim, output_dim, hidden_size, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []

        # First layer: input_dim -> hidden_size per head
        layers.append(nn.Conv1d(output_dim * input_dim, output_dim * hidden_size,
                                kernel_size=1, groups=output_dim))
        layers.append(nn.ELU())

        # Intermediate layers: hidden_size -> hidden_size per head
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(output_dim * hidden_size, output_dim * hidden_size,
                                    kernel_size=1, groups=output_dim))
            layers.append(nn.ELU())

        # Final layer: hidden_size -> 1 per head
        layers.append(nn.Conv1d(output_dim * hidden_size, output_dim,
                                kernel_size=1, groups=output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 1:
            x = x.view(1, -1)

        B = x.shape[0]

        # tile input output_dim times, one copy per head
        # (B, input_dim) -> (B, output_dim * input_dim, 1)
        x_expanded = x.unsqueeze(1).expand(B, self.output_dim, self.input_dim)
        x_expanded = x_expanded.reshape(B, self.output_dim * self.input_dim, 1)

        out = self.net(x_expanded)  # (B, output_dim, 1)
        return out.squeeze(-1)      # (B, output_dim)


def train_DebReg(model, optimizer, train_loader, val_loader, num_epochs, scheduler, early_stop_patience):
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

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
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




