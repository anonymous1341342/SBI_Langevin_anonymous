import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import ot
# from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# No prior boundary, we use log_theta

# Settings for the floor and room assignments
K = 10 # number of floors
N = 600
NR = 2 # number of people in each room
NF = int(N/K) # number of people on each floor

F_assign = torch.zeros(N, K)
for k in range(K):
    F_assign[(k*NF):((k+1)*NF), k] = 1
C_F = F_assign @ F_assign.T 

R_assign = torch.zeros(N, int(N/NR))
for r in range( int(N/NR) ):
    R_assign[(r*NR):((r+1)*NR), r] = 1
C_R = R_assign @ R_assign.T

F_assign = F_assign.to(device)
C_F = C_F.to(device)
C_R = C_R.to(device)

gamma = 0.05
alpha = 0.1
eta = 0.1 
T = 52

#####################
#  Data Generation  #
#####################
def Ind(t):
    """
        Indicator function
    """
    return ( 1.0 * (t > 0) )

def m_vec_partial(N, T, beta, gamma, alpha, eta, F_assign, C_F, C_R, Z, NF, NR):
    """
        generate data in the fully observed case (Algorithm 3 in the paper)
    """
    # eta is the probability that the infection has symptons (can be observed)
    # Z = [allD, allU, allV]: latent variables, allD of dimension N by T is indicator of replacement
    # allU of dimension N by T is for bernoulli sampling, allV of dimension N by T is for bernoulli sampling
    allD, allU, allV = Z
    allD, allU, allV = allD.to(device), allU.to(device), allV.to(device)
    X = torch.zeros(N, T).to(device)
    Y = torch.zeros(N, T).to(device)
    X[:, 0] = Ind(alpha - allU[:, 0]) # initialization
    Y[:, 0] = X[:, 0] # patients are screened when they enter
    for t in range(1, T):
        # First, update X
        D = allD[:, t] # discharge or not
        X[D == 1, t] = Ind(alpha - allU[D == 1, t]) # for replaced patients
        norep_infec_id = torch.logical_and(D == 0, X[:, t-1] > 0.5) # people who are not replaced and have already been infected
        norep_sus_id = torch.logical_and(D == 0, X[:, t-1] < 0.5) # people who are not replaced and are not infected

        X[norep_infec_id, t] = 1 # X[norep_infec_id, t-1]
        lam = hazard(beta, X[:, t-1], F_assign, C_F, C_R, N, NF, NR)
        lam = lam[norep_sus_id]
        X[norep_sus_id, t] = Ind( (1 - (-lam).exp()) - allU[norep_sus_id, t] )

        # Next, get Y based on X
        Y[D == 1, t] = X[D == 1, t]
        id1 = torch.logical_and(X[:, t] > 0.5, Y[:, t-1] < 0.5)
        Y[torch.logical_and(D == 0, id1), t] = Ind(eta - allV[torch.logical_and(D == 0, id1), t])
        Y[torch.logical_and(D == 0, ~id1), t] = Y[torch.logical_and(D == 0, ~id1), t - 1]
    return Y # , X

def hazard(beta, X, F_assign, C_F, C_R, N, NF, NR):
    """
    Calculate the hazard function based on the previous state X_{t-1} and the contact matrices C_F and C_R
    """
    # Input:
    # beta0, beta_middle, beta_last: the parameters, beta_middle is a K-dimensional vector, the other two are scalars
    # X: X_{t-1}, a N dimensional vector
    # F_assign: assignment of floor, a N by K matrix, F_assign[i, k] = individual i lives on floor k
    # C_F: contact matrix of floor, C_F[i, j] = 1{i and j are on the same floor}
    # C_R: contact matrix of room, C_R[i, j] = 1{i and j are in the same room}
    # C_F can be calculated from F_assign, but we make them input to save calculation, cause they are unchanged throughout the dynamics
    # N, NF, NR: scale factors for beta
    # Output:
    # lambda: lambda(t) = (lambda_1(t), ..., lambda_N(t)), recording the hazard for each individual
    N = X.shape[0]
    beta0 = beta[0] / N
    beta_middle = beta[1:-1] / NF
    beta_last = beta[-1] / NR

    return ( beta0 * torch.ones(N, N).to(device) + (F_assign @ beta_middle).view(-1, 1).repeat(1, N) * C_F + beta_last * C_R ) @ X
    # return ( beta0 * torch.ones(N, N) + beta_last * C_R ) @ X

def gen_z(N, T):
    """
        generate the latent random variable
    """
    return [torch.bernoulli(gamma * torch.ones(N, T)).to(device), torch.rand(N, T).to(device), torch.rand(N, T).to(device)]

#######################################################
#               Generate reference table              #
#######################################################
def get_SS(y):
    """
        get the 364-dimensional summary statistics
    """
    all_rate = y.mean(dim = 0)
    floor_rates = y.reshape(K, NF, T).mean(dim = 1)
    room_rate = ( (y.reshape(-1, NR, T).sum(dim = 1) > 1.5) * 1.0 ).mean(dim = 0) # rate that both roommates are infected
    return torch.cat( (all_rate, floor_rates.ravel(), room_rate), dim = 0 )

def gen_ref_log(mean_theta, std_theta, sample_size = 10000):
    """
        generate theta from a (truncated) gaussian proposal distribution, and then use theta to generate x
        mean_theta: the mean of the truncated normal, a 1-dim tensor of the same length as theta
        std_theta: the std of the truncated normal, a 1-dim tensor of the same length as theta
        lower: the lower bound for each dimension of theta, a 1-dim tensor
        upper: the upper bound for each dimension of theta, a 1-dim tensor
    """
    
    mu_new = mean_theta.view(1, -1).repeat(sample_size, 1)
    sigma_new = std_theta.view(1, -1).repeat(sample_size, 1)

    # draw theta_r0
    log_theta = mu_new + sigma_new * torch.randn(mu_new.shape).to(device)
    log_theta = log_theta.to(device)
    theta = log_theta.exp()
    
    # draw data_r0
    data = torch.zeros(sample_size, (K+2) * T).to(device)
    for i in range(sample_size):
        z = gen_z(N, T)
        y = m_vec_partial(N, T, theta[i], gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
        data[i] = get_SS(y)
    return log_theta, data

#############################################
#               Score Matching              #
#############################################
# ELU() activation
# no additive structure
class ELU_Nonadd_Large(nn.Module):
    def __init__(self, input_size, output_size):
        super(ELU_Nonadd_Large, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 800),
            nn.ELU(),
            
            nn.Linear(800, 512),
            nn.ELU(),
            
            nn.Linear(512, 256),
            nn.ELU(),

            nn.Linear(256, 128),
            nn.ELU(),

            nn.Linear(128, output_size)
        )

    def forward(self, log_theta, x):
        if len(log_theta.shape) == 1: # if one-dimensional
            log_theta = log_theta.view(-1, 1)

        if len(x.shape) == 1:
            x = x.view(-1, 1)
        score = self.layers(torch.cat((log_theta, x), dim = 1))
        return score

# ELU() activation
# no additive structure
class ELU_Nonadd_Medium(nn.Module):
    def __init__(self, input_size, output_size):
        super(ELU_Nonadd_Medium, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ELU(),
            
            nn.Linear(512, 256),
            nn.ELU(),

            nn.Linear(256, 128),
            nn.ELU(),

            nn.Linear(128, output_size)
        )

    def forward(self, log_theta, x):
        if len(log_theta.shape) == 1: # if one-dimensional
            log_theta = log_theta.view(-1, 1)

        if len(x.shape) == 1:
            x = x.view(-1, 1)
        score = self.layers(torch.cat((log_theta, x), dim = 1))
        return score

# ELU() activation
# no additive structure
class ELU_Nonadd_Small(nn.Module):
    def __init__(self, input_size, output_size):
        super(ELU_Nonadd_Small, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ELU(),
            
            nn.Linear(256, 128),
            nn.ELU(),

            nn.Linear(128, 64),
            nn.ELU(),

            nn.Linear(64, output_size)
        )

    def forward(self, log_theta, x):
        if len(log_theta.shape) == 1: # if one-dimensional
            log_theta = log_theta.view(-1, 1)

        if len(x.shape) == 1:
            x = x.view(-1, 1)
        score = self.layers(torch.cat((log_theta, x), dim = 1))
        return score

# ELU() activation
# no additive structure
class ELU_Nonadd_XSmall(nn.Module):
    def __init__(self, input_size, output_size):
        super(ELU_Nonadd_XSmall, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ELU(),
            
            nn.Linear(128, 128),
            nn.ELU(),

            nn.Linear(128, 64),
            nn.ELU(),

            nn.Linear(64, output_size)
        )

    def forward(self, log_theta, x):
        if len(log_theta.shape) == 1: # if one-dimensional
            log_theta = log_theta.view(-1, 1)

        if len(x.shape) == 1:
            x = x.view(-1, 1)
        score = self.layers(torch.cat((log_theta, x), dim = 1))
        return score





def Like_score_loss5(model, theta, x, prop_score, g, g1):
    # the same as loss4 function
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

def train_deb5(model, optimizer, dataloader, val_dataloader, g, g1, num_epochs, scheduler, early_stop_patience):
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
            loss, bias = Like_score_loss5(model, batch_theta, batch_x, batch_prop_score, g, g1)
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
            val_loss, val_bias = Like_score_loss5(model, val_batch_theta, val_batch_x, val_batch_prop_score, g, g1)
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
        loss, bias = Like_score_loss5(model, batch_theta, batch_x, batch_prop_score, g, g1)
        total_bias += bias.detach()
    # with torch.no_grad(): 
    #     model.layers[-1].bias -= (total_bias / len(dataloader)).to(device) 

    bias_lastlayer = total_bias / len(dataloader)
    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total training time: {round(total_duration, 2)} seconds')
    return bias_lastlayer







# h(t) = t, identity map
def g(theta, x):
    return torch.ones(theta.shape).to(device)

def g1(theta, x):
    return torch.zeros(theta.shape).to(device)


