import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import copy
from itertools import cycle
from torch.autograd.functional import jacobian
import scipy
from scipy.optimize import brentq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 0.1
M = 10

# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0


#############################################################
#         function to generate the Bernstein basis          #
#############################################################
 
# function to generate the Bernstein basis
def get_psi(x, M):
    """
    Input:
        x: a 1-d tensor
        M: a positive integer
    Output:
        Bernstein polynomial terms, a matrix of dimension x.shape[0] * M + 1
    """
    psi = torch.zeros(x.shape[0], M + 1)
    for k in range(M + 1):
        psi[:, k] = math.comb(M, k) * x**k * (1 - x)**(M - k)
    return psi

def get_A(M):
    """
        get the matrix A, for a given degree M
    """
    A = torch.eye(M + 1)
    for i in range(1, M + 1):
        A[i, i - 1] = -1
    return A

######################################################################
#        Score matching using a truncated normal distribution        #
######################################################################

def gen_ref(mean_theta, std_theta, lower, upper, obs_size, sample_size = 10000):
    """
        generate theta from a (truncated) gaussian proposal distribution, and then use theta to generate x
        mean_theta: the mean of the truncated normal, a 1-dim tensor of the same length as theta
        std_theta: the std of the truncated normal, a 1-dim tensor of the same length as theta
        lower: the lower bound for each dimension of theta, a 1-dim tensor
        upper: the upper bound for each dimension of theta, a 1-dim tensor
    """

    
    # use inverse sampling to draw theta from truncated normal
    dist = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)
    mu_new = mean_theta.view(1, -1).repeat(sample_size, 1)
    sigma_new = std_theta.view(1, -1).repeat(sample_size, 1)
    lower_bound = lower.view(1, -1).repeat(sample_size, 1)
    upper_bound = upper.view(1, -1).repeat(sample_size, 1)

    theta_time1 = time.time()
    # draw theta_r0
    uni = torch.rand(sample_size, M + 1) # .to(device)
    theta = mu_new + sigma_new * dist.icdf( (1 - uni) * dist.cdf((lower_bound - mu_new) / sigma_new) + uni * dist.cdf((upper_bound - mu_new) / sigma_new) )
    theta_time2 = time.time()
    # print(f"time of generating theta: {round(theta_time2 - theta_time1)} seconds")
    

    x = torch.rand(sample_size, obs_size) # .to(device)
    y = torch.zeros(sample_size, obs_size) # .to(device)
    A = get_A(M) # .to(device)
    for i in range(sample_size):
        # get design matrix of x[i]
        psi = get_psi(x[i], M) # .to(device)
        # generate y[i]
        y[i] = psi @ torch.linalg.inv(A) @ theta[i] + sigma * torch.randn(obs_size) # .to(device)
    
    # data = torch.stack((x, y), dim=2).reshape(x.shape[0], -1)
    data = torch.zeros(sample_size, 2 * obs_size)
    data[:, ::2] = x
    data[:, 1::2] = y
    
    return theta, data

def gen_ref_distinct_theta(mean_theta, std_theta, lower, upper, sample_size = 10000):
    """
        generate theta from a (truncated) gaussian proposal distribution, and then use theta to generate x
        mean_theta: the mean of the truncated normal, a 1-dim tensor of the same length as theta
        std_theta: the std of the truncated normal, a 1-dim tensor of the same length as theta
        lower: the lower bound for each dimension of theta, a 1-dim tensor
        upper: the upper bound for each dimension of theta, a 1-dim tensor
    """
    time_start = time.time()
    
    # use inverse sampling to draw theta from truncated normal
    dist = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)
    mu_new = mean_theta.view(1, -1).repeat(sample_size, 1)
    sigma_new = std_theta.view(1, -1).repeat(sample_size, 1)
    lower_bound = lower.view(1, -1).repeat(sample_size, 1)
    upper_bound = upper.view(1, -1).repeat(sample_size, 1)

    theta_time1 = time.time()
    # draw theta_r0
    uni = torch.rand(sample_size, M + 1) # .to(device)
    theta = mu_new + sigma_new * dist.icdf( (1 - uni) * dist.cdf((lower_bound - mu_new) / sigma_new) + uni * dist.cdf((upper_bound - mu_new) / sigma_new) )
    theta_time2 = time.time()
    print(f"time of generating theta: {round(theta_time2 - theta_time1)} seconds")


    x = torch.rand(sample_size)
    A = get_A(M)
    psi = get_psi(x, M) # (sample_size, M + 1)
    y = ( (psi @ torch.linalg.inv(A)) * theta ).sum(dim = 1) + sigma * torch.randn(sample_size)
    
    
    data = torch.zeros(sample_size, 2)
    data[:, 0] = x
    data[:, 1] = y

    time_end = time.time()
    print(f"Total time for generating ABC table = {round((time_end - time_start) / 60, 3)} minutes")
    return theta, data

# get the inf_rate
def get_inf_rate(mode, std_orig, lower, upper, actual_inf_rate):
    """
    Find res such that the truncated normal N(mode, (res * std_orig)^2) on (lower, upper)
    has standard deviation = actual_inf_rate * std_orig

    Parameters:
        mode: float, mean of the untruncated normal (also the mode of truncated)
        std_orig: float, original std of untruncated normal
        lower: float, lower truncation bound
        upper: float, upper truncation bound
        actual_inf_rate: float, desired std increase ratio for the truncated distribution

    Returns:
        res: float, the inflation factor to multiply std_orig
    """
    target_std = actual_inf_rate * std_orig

    # cannot exceed the variance of a uniform distribution
    if target_std**2 > (upper - lower)**2 / 12:
        return 5

    def truncated_std(sigma):
        """Compute std of truncated normal N(mode, sigma^2) on (lower, upper)"""
        a = (lower - mode) / sigma
        b = (upper - mode) / sigma
        Z = scipy.stats.norm.cdf(b) - scipy.stats.norm.cdf(a)
        phi_a = scipy.stats.norm.pdf(a)
        phi_b = scipy.stats.norm.pdf(b)
        m = (phi_a - phi_b) / Z
        var = sigma**2 * (1 + (a * phi_a - b * phi_b) / Z - m**2)
        return var**0.5

    # Define root function in terms of res
    def f(res):
        return truncated_std(res * std_orig) - target_std

    try:
        res_opt = brentq(f, 1, 20, xtol=1e-6)
    except ValueError:
        raise RuntimeError("Unable to find inflation factor")

    return res_opt

class single_ELU_LikeScoreMatchingNN(nn.Module):
    def __init__(self, theta_dim, x_dim, obs_size, hidden_size, num_layers = 1):
        super(single_ELU_LikeScoreMatchingNN, self).__init__()
        
        layers = [nn.Linear(theta_dim + x_dim, hidden_size), nn.ELU()]

        # Add hidden layers based on num_layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ELU())

        # Output layer to match the desired output size
        layers.append(nn.Linear(hidden_size, theta_dim))

        # Combine all layers into a sequential model
        self.layers = nn.Sequential(*layers)
        self.obs_size = obs_size
        self.x_dim = x_dim

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


# # returns the distance of \theta_j to the boundary when fixing (\theta_{-j}, x)
def dist2bd(theta, x):
    # input of theta and x can have n rows, each row is an observation

    lower = a * torch.ones(theta.shape).to(device)
    lower[:, 0] = a0 # theta0 has different support
    
    upper = b * torch.ones(theta.shape).to(device)
    upper[:, 0] = b0 # theta0 has different support
    
    return torch.min(theta - lower, upper - theta), lower, upper


def Like_score_loss(model, theta, x, prop_score, g, g1):
    # g: weight function, takes input (theta, x), output dimension is the same as theta
    # g1: first derivative of g (the diagonal part, \partial g(\theta, x)_j / \partial \theta_j), output dimension is the same as theta
    # we require g and g1 to be able to address matrix input and do element-wise mapping
    bias = model(theta, x).mean(dim = 0)
    score = model(theta, x) - bias
    loss1 = torch.norm(score * g(theta, x)**(1/2), dim = -1) ** 2 / 2.
    loss3 = ((score * g(theta, x)) * prop_score).sum(dim = -1)
    
    theta.requires_grad_(True)
    score_tmp = model(theta, x) # In order to calculate grad2
    loss2 = torch.zeros(theta.shape[0]).to(device)
    for i in range(theta.shape[1]):
        grad2 = torch.autograd.grad(outputs = score_tmp[:, i].sum(), inputs = theta, create_graph=True)[0][:, i]
        loss2 += grad2 * (g(theta, x)[:, i]) + score[:, i] * g1(theta, x)[:, i]
    
    loss = loss1 + loss2 + loss3
    return loss.mean(), bias

def train_deb(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience):
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None

    # record training loss and validation loss at each epoch and then plot
    start_time = time.time()
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train() 
        total_loss = 0.0
        valid_batches = 0
        for batch_sample in dataloader:
            optimizer.zero_grad()
            batch_theta, batch_x, batch_prop_score = batch_sample
            batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
            loss, bias = Like_score_loss(model, batch_theta, batch_x, batch_prop_score, g, g1)
            if torch.isnan(loss):
                print(f"[WARNING] NaN detected, skipping this minibatch")
                continue
            valid_batches += 1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / valid_batches

        model.eval()
        total_loss_val = 0.0
        val_valid_batches = 0
        for val_batch_sample in val_dataloader:
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_sample
            val_batch_theta, val_batch_x, val_batch_prop_score = val_batch_theta.to(device), val_batch_x.to(device), val_batch_prop_score.to(device)
            val_loss, val_bias = Like_score_loss(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
            if torch.isnan(val_loss):
                # print(f"[WARNING] NaN detected, skipping this minibatch")
                continue
            val_valid_batches += 1
            total_loss_val += val_loss.item()    

        avg_val_loss = total_loss_val / val_valid_batches
        if avg_val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                print(f"Epoch {epoch+1}: reducing learning rate to {new_lr:.2e}")
        
        time2 = time.time()
        if epoch % 1 == 0 or epoch == num_epochs:
            print(f'Epoch {epoch+1}/{num_epochs} | Training Loss: {round(avg_loss, 3)} | Validation Loss: {round(avg_val_loss, 3)} | Time: {round(time2 - time1, 2)} seconds')

        # early stop
        if (epoch+1) - best_epoch >= early_stop_patience:
            print(f"Val_loss didn't improve after {early_stop_patience} epochs, stop training")
            break

    # Load best model state after training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        print(f"Return the best model at epoch {best_epoch}, with Validation Loss: {best_val_loss:.3f}")

    
    # output the final model, we just need to minus the bias
    # we calculate the bias using the whole dataset
    total_bias = 0.0 # is actually a vector of the same dimension as theta
    for batch_sample in dataloader:
        batch_theta, batch_x, batch_prop_score = batch_sample
        batch_theta, batch_x, batch_prop_score = batch_theta.to(device), batch_x.to(device), batch_prop_score.to(device)
        loss, bias = Like_score_loss(model, batch_theta, batch_x, batch_prop_score, g, g1)
        total_bias += bias.detach()
    # with torch.no_grad(): 
    #     model.layers[-1].bias -= (total_bias / len(dataloader)).to(device) 

    bias_lastlayer = total_bias / len(dataloader)
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total training time: {round(total_duration, 2)} seconds')
    return bias_lastlayer



def Fisher_penalty(model, theta_extra, x_extra):
    # the same as loss4 function
    # g: weight function, takes input (theta, x), output dimension is the same as theta
    # g1: first derivative of g (the diagonal part, \partial g(\theta, x)_j / \partial \theta_j), output dimension is the same as theta
    # we require g and g1 to be able to address matrix input and do element-wise mapping

    def score_sum_fn(theta_): # in order to calculate Jacobian
        return ( model.cal_penalty(theta_, x_extra).sum(dim = 1) ).sum(dim = 0)
    
    theta_extra.requires_grad_(True)
    obs_size = int(x_extra.shape[1] / model.x_dim) # extra_obs_size
    E_Jacobian = jacobian(score_sum_fn, theta_extra, create_graph = True).permute(1, 0, 2) / obs_size # estimated mean of the Jacobian, of dimension (batch_size, theta_dim, theta_dim)

    score_tensor_extra = model.cal_penalty(theta_extra, x_extra)
    bias_extra_single = ( score_tensor_extra.sum(dim = 1).mean(dim = 0) / obs_size ).unsqueeze(0).unsqueeze(0)
    EssT = torch.einsum('ibc,ibd->icd', score_tensor_extra - bias_extra_single, score_tensor_extra - bias_extra_single) / obs_size 
    # EssT = torch.einsum('ibc,ibd->icd', score_tensor_extra, score_tensor_extra) / obs_size 
    penalty_fisher = ( (EssT + E_Jacobian)**2 ).sum(dim = (1, 2)) # a vector storing the Fnorm^2 for every row in the minibatch
    
    return penalty_fisher.mean()


def cal_ssT(model, theta_extra, x_extra):
    # calculate E[||E(ssT)||_F^2]

    score_tensor_extra = model.cal_penalty(theta_extra, x_extra)
    obs_size = int(x_extra.shape[1] / model.x_dim) # extra_obs_size
    bias_extra_single = ( score_tensor_extra.sum(dim = 1).mean(dim = 0) / obs_size ).unsqueeze(0).unsqueeze(0)
    EssT = torch.einsum('ibc,ibd->icd', score_tensor_extra - bias_extra_single, score_tensor_extra - bias_extra_single) / obs_size 
    
    return ( EssT**2 ).sum(dim = (1, 2)).mean()

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



def Deb_curve(model, x):
    # get the jacobian of the DebReg model

    def reg_output(x_): # in order to calculate Jacobian
        return ( model(x_) ).sum(dim = 0)
    
    x.requires_grad_(True)
    # jacob is for the minibatch data, has dim (b, d, d)
    jacob = jacobian(reg_output, x, create_graph = True).permute(1, 0, 2)
    
    return jacob

# The loss is E_\theta{||g - Es||_F^2 + \lam_curve * ||gg^T + \nabla g||_F^2}
def train_DebReg_fisher_simpleversion(model, optimizer, train_loader, val_loader, num_epochs, lam_curve):
    model.to(device)
    loss_fn = nn.MSELoss()

    best_val_reg_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None
    best_epoch = None

    start_time = time.time()
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        total_obj = 0.0
        total_reg_loss = 0.0
        total_curve_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            pred = model(xb)
            pred_predT = pred.unsqueeze(2) @ pred.unsqueeze(1)
            jacob = Deb_curve(model, xb)
            loss_reg = loss_fn(pred, yb)
            loss_curve = loss_fn(pred_predT, -jacob)
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
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            pred_predT = pred.unsqueeze(2) @ pred.unsqueeze(1)
            jacob = Deb_curve(model, xb)
            loss_reg = loss_fn(pred, yb)
            loss_curve = loss_fn(pred_predT, -jacob)
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
        print(f"Epoch {epoch+1} | Train Loss (Total, Reg, Curve): ({avg_obj:.4f}, {avg_reg_loss:.4f},{avg_curve_loss:.4f}) | Val Loss (Total, Reg, Curve): ({avg_val_obj:.4f}, {avg_val_reg_loss:.4f},{avg_val_curve_loss:.4f}) | Time: {round(time2 - time1, 2)} seconds")


    print(f"Return the model at epoch {best_epoch}, with validation Reg loss {best_val_reg_loss:.4f}")
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    end_time = time.time()
    print(f"Total training time of the DebReg_curve model = {(end_time - start_time)/60:.2f} minutes")


# The loss is E_\theta{||g - Es||_F^2 + \lam_curve * ||gg^T - \nabla g - Es g^T - g Es^T||_F^2}
def train_DebReg_fisher_crossterm(model, optimizer, train_loader, val_loader, num_epochs, lam_curve, best_val_reg_loss):
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

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            pred = model(xb)
            pred_predT = pred.unsqueeze(2) @ pred.unsqueeze(1)
            jacob = Deb_curve(model, xb)
            loss_reg = loss_fn(pred, yb)
            loss_curve = loss_fn( pred_predT, jacob + torch.einsum('bi,bj->bij', yb, pred) + torch.einsum('bi,bj->bij', pred, yb) )
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
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            pred_predT = pred.unsqueeze(2) @ pred.unsqueeze(1)
            jacob = Deb_curve(model, xb)
            loss_reg = loss_fn(pred, yb)
            loss_curve = loss_fn( pred_predT, jacob + torch.einsum('bi,bj->bij', yb, pred) + torch.einsum('bi,bj->bij', pred, yb) )
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



