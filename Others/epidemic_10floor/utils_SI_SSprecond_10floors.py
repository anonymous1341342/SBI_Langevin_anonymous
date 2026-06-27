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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

F_assign = F_assign.to(torch.float64).to(device)
C_F = C_F.to(torch.float64).to(device)
C_R = C_R.to(torch.float64).to(device)

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
    return ( 1.0 * (t > 0) ).to(torch.float64)

def m_vec_partial(N, T, beta, gamma, alpha, eta, F_assign, C_F, C_R, Z, NF, NR):
    """
        generate data in the fully observed case (Algorithm 3 in the paper)
    """
    # eta is the probability that the infection has symptons (can be observed)
    # Z = [allD, allU, allV]: latent variables, allD of dimension N by T is indicator of replacement
    # allU of dimension N by T is for bernoulli sampling, allV of dimension N by T is for bernoulli sampling
    allD, allU, allV = Z
    allD, allU, allV = allD.to(device), allU.to(device), allV.to(device)
    X = torch.zeros(N, T, dtype = torch.float64).to(device)
    Y = torch.zeros(N, T, dtype = torch.float64).to(device)
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

def soft_Ind(t):
    """
        Indicator function
    """
    return 1 / ( 1 + (-500.0 * t).exp() )

def soft_m_vec_partial(N, T, beta, gamma, alpha, eta, F_assign, C_F, C_R, Z, NF, NR):
    """
        generate data in the fully observed case (Algorithm 3 in the paper)
    """
    # eta is the probability that the infection has symptons (can be observed)
    # Z = [allD, allU, allV]: latent variables, allD of dimension N by T is indicator of replacement
    # allU of dimension N by T is for bernoulli sampling, allV of dimension N by T is for bernoulli sampling
    allD, allU, allV = Z
    allD, allU, allV = allD.to(device), allU.to(device), allV.to(device)
    X = torch.zeros(N, T, dtype = torch.float64).to(device)
    Y = torch.zeros(N, T, dtype = torch.float64).to(device)
    X[:, 0] = soft_Ind(alpha - allU[:, 0]) # initialization
    Y[:, 0] += X[:, 0] # patients are screened when they enter
    for t in range(1, T):
        # First, update X
        D = allD[:, t] # discharge or not
        X[D == 1, t] = soft_Ind(alpha - allU[D == 1, t]) # for replaced patients
        norep_infec_id = torch.logical_and(D == 0, X[:, t-1] > 0.5) # people who are not replaced and have already been infected
        norep_sus_id = torch.logical_and(D == 0, X[:, t-1] < 0.5) # people who are not replaced and are not infected

        X[norep_infec_id, t] = X[norep_infec_id, t-1] # 1
        lam = hazard(beta, X[:, t-1].clone(), F_assign, C_F, C_R, N, NF, NR)
        lam = lam[norep_sus_id]
        X[norep_sus_id, t] = soft_Ind( (1 - (-lam).exp()) - allU[norep_sus_id, t] )

        # Next, get Y based on X
        Y[D == 1, t] = X[D == 1, t]
        id1 = torch.logical_and(X[:, t] > 0.5, Y[:, t-1] < 0.5)
        # Y[torch.logical_and(D == 0, id1), t] = soft_Ind(eta - allV[torch.logical_and(D == 0, id1), t])
        Y[torch.logical_and(D == 0, id1), t] = soft_Ind(eta - allV[torch.logical_and(D == 0, id1), t]) * X[torch.logical_and(D == 0, id1), t]
        Y[torch.logical_and(D == 0, ~id1), t] = Y[torch.logical_and(D == 0, ~id1), t - 1]
    # obs_case = 1.0 * ( (X - Y).abs() < 0.1 )
    return Y # X * obs_case

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

    return ( beta0 * torch.ones(N, N, dtype = torch.float64).to(device) + (F_assign @ beta_middle).view(-1, 1).repeat(1, N) * C_F + beta_last * C_R ) @ X
    # return ( beta0 * torch.ones(N, N) + beta_last * C_R ) @ X

#####################
#  Preconditioning  #
#####################
def W1_1d_vec(x, y):
    """
        Vectorized version of calculating W_2^2 for many pairs of (x_i, y_i)
        x: a n_x by n_u tensor
        y: a n_y by n_u tensor

        Output: a 1 by n_u tensor, each element records the W_1 distance between the two corresponding x column and y column
    """
    device = x.device # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nx = x.shape[0]
    ny = y.shape[0]
    nu = x.shape[1]

    # order x and y respectively, from the smallest to the largest
    x = x.sort(dim = 0).values
    y = y.sort(dim = 0).values

    # get the union of the partition points
    t_set = torch.cat((1/nx * torch.arange(0, nx + 1), 1/ny * torch.arange(0, ny + 1))).unique().sort().values.to(device)

    # get F_X^{-1}(t0), F_X^{-1}(t1), ..., F_X^{-1}(tN)
    idx = (nx * t_set).ceil() - 1 # the indices of x, -1 because python starts from 0
    Fx_inverse_set = x[idx.to(torch.int), :]
    # get F_Y^{-1}(t0), F_Y^{-1}(t1), ..., F_Y^{-1}(tN)
    idy = (ny * t_set).ceil() - 1
    Fy_inverse_set = y[idy.to(torch.int), :]
    # diff_set = |F_X^{-1}(t0) - F_Y^{-1}(t0), ..., F_X^{-1}(tN) - F_Y^{-1}(tN)|
    diff_set = (Fx_inverse_set - Fy_inverse_set).abs()

    # Now we can calculate the final result!
    return ( (t_set[1:] - t_set[:-1]).view(-1, 1).repeat(1, nu) * diff_set[1:] ).sum(dim = 0)

def samen_W1_1d_vec(x, y):
    """
        When x and y have the same sample size
    
        Vectorized version of calculating W_2^2 for many pairs of (x_i, y_i)
        x: a n by n_u tensor
        y: a n by n_u tensor

        Output: a 1 by n_u tensor, each element records the W_1 distance between the two corresponding x column and y column
    """
    device = x.device # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # order x and y respectively, from the smallest to the largest
    x = x.sort(dim = 0).values
    y = y.sort(dim = 0).values

    # Now we can calculate the final result!
    return (x - y).abs().mean(dim = 0)

def samen_W2_squared_1d_vec(x, y):
    """
        Vectorized version of calculating W_2^2 for many pairs of (x_i, y_i)
        x: a n by n_u tensor
        y: a n by n_u tensor

        Output: a 1 by n_u tensor, each element records the W_2^2 distance between the two corresponding x column and y column
    """
    device = x.device # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # order x and y respectively, from the smallest to the largest
    x = x.sort(dim = 0).values
    y = y.sort(dim = 0).values

    return ( (x - y)**2 ).mean(dim = 0)

def gen_u(usize, udim):
    """
        Draw u from the Uniform distribution on the surface of the unit-L2ball
    """
    xi = torch.randn(usize, udim, dtype = torch.float64) # draw multivariate gaussian
    return xi / torch.linalg.norm(xi, dim = 1).view(-1, 1).repeat(1, udim)

def gen_z(N, T):
    """
        generate the latent random variable
    """
    return [torch.bernoulli(gamma * torch.ones(N, T)).to(torch.float64).to(device), torch.rand(N, T).to(torch.float64).to(device), torch.rand(N, T).to(torch.float64).to(device)]

def cm_SS_log_Adam_SW_fixz(lam_time, metric, x_obs, u_size, z, log_theta_init, lr = 0.1, maxiter = 100, plot = True):
    """
    x_obs: observed x
    simu_size: the size of simulated data
    theta_init: initial value of theta
    lr: learning rate (step size) of gradient descent
    lam_time: lam determines the importance of matching the time t

    We use full batch!
    """

    x_obs = x_obs.to(device)
    x_obs = torch.cat([x_obs, lam_time * torch.arange(1, 53).view(-1, 1).to(torch.float64).to(device)], dim=1)
    log_theta = log_theta_init.to(device)
    log_theta.requires_grad_(True)
    optimizer = optim.Adam([log_theta], lr=lr)
    
    log_theta_path = [] # store the results
    train_loss_path = []

    # marginal distribution
    for iter in range(maxiter):
        optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        y_simu = soft_m_vec_partial(N, T, log_theta.exp(), gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
        y_simu = soft_get_SS(y_simu).reshape(-1, 52).T # the summary statistics
        y_simu = torch.cat([y_simu, lam_time * torch.arange(1, 53).view(-1, 1).to(torch.float64).to(device)], dim=1)
        # print(y_simu)
        u = gen_u(u_size, x_obs.shape[1]).to(device)
        x_obs_projected = ( u.unsqueeze(0).repeat(x_obs.shape[0], 1, 1) * x_obs.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)        
        y_simu_projected = ( u.unsqueeze(0).repeat(y_simu.shape[0], 1, 1) * y_simu.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)
        
        if metric == "squaredW2":
            if x_obs.shape[0] == y_simu.shape[0]:
                loss = samen_W2_squared_1d_vec(x_obs_projected, y_simu_projected).mean()
            else:
                loss = W2_squared_1d_vec(x_obs_projected, y_simu_projected).mean()
        if metric == "W1":
            if x_obs.shape[0] == y_simu.shape[0]:
                loss = samen_W1_1d_vec(x_obs_projected, y_simu_projected).mean()
            else:            
                loss = W1_1d_vec(x_obs_projected, y_simu_projected).mean()
        if metric == "naive": # bad
            loss = torch.linalg.norm(y_simu - x_obs)
        
        loss.backward()
        # theta.grad = torch.nan_to_num(theta.grad, nan=0.0)
        # print("gradient:", theta.grad)
        # print("theta:", theta.detach())
        optimizer.step()
        
        # no constraint 
        
        log_theta_path.append(log_theta.detach().clone())
        train_loss_path.append(loss.item())

    if plot == True:
        # Plot the training loss
        plt.figure(figsize=(8, 4))
        plt.plot(range(maxiter), train_loss_path, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Over Iterations')
        plt.show()
    return log_theta_path


def cm_SS_log_res_diffz(lam_time, metric, data_obs, simu_size, num_diffz, lr, maxiter):
    res_diffz = torch.zeros(num_diffz, 12, dtype = torch.float64)
    
    u_size = 100
    for i in range(num_diffz):
        # generate z
        z = gen_z(simu_size, T)
        
        # random initialization
        theta_init = 0.5 * torch.rand(12, dtype = torch.float64)
        log_theta_init = theta_init.log()
    
        # solve the SW objective by Adam
        log_theta_path = cm_SS_log_Adam_SW_fixz(lam_time, metric, data_obs, u_size, z, log_theta_init, lr, maxiter, plot = False)
    
        # record the solution
        res_diffz[i] = log_theta_path[-1]
    return res_diffz

def soft_Ind2(t):
    """
        Indicator function
    """
    return 1 / ( 1 + (-100.0 * t).exp() )

def soft_get_SS(y):
    """
        get the 364-dimensional summary statistics
    """
    all_rate = y.mean(dim = 0)
    floor_rates = y.reshape(K, NF, T).mean(dim = 1)
    room_rate = ( soft_Ind2(y.reshape(-1, NR, T).sum(dim = 1) - 1.5) ).mean(dim = 0)
    # room_rate = ( (y.reshape(-1, NR, T).sum(dim = 1) > 1.5) * 1.0 ).mean(dim = 0) # rate that both roommates are infected
    return torch.cat( (all_rate, floor_rates.ravel(), room_rate), dim = 0 )

def get_SS(y):
    """
        get the 364-dimensional summary statistics
    """
    all_rate = y.mean(dim = 0)
    floor_rates = y.reshape(K, NF, T).mean(dim = 1)
    room_rate = ( (y.reshape(-1, NR, T).sum(dim = 1) > 1.5) * 1.0 ).mean(dim = 0) # rate that both roommates are infected
    return torch.cat( (all_rate, floor_rates.ravel(), room_rate), dim = 0 )


# delay reconstruction
def delay_SS_log_Adam_SW_fixz(delay, metric, x_obs, u_size, z, log_theta_init, lr = 0.1, maxiter = 100, plot = True):
    """
    x_obs: observed x
    simu_size: the size of simulated data
    theta_init: initial value of theta
    lr: learning rate (step size) of gradient descent
    lam_time: lam determines the importance of matching the time t

    We use full batch!
    """

    x_obs = x_obs.to(device)
    x_obs = torch.cat([x_obs[i:(i+delay)].reshape(1, -1) for i in range(len(x_obs) - delay + 1)], dim=0)
    log_theta = log_theta_init.to(device)
    log_theta.requires_grad_(True)
    optimizer = optim.Adam([log_theta], lr=lr)
    
    log_theta_path = [] # store the results
    train_loss_path = []

    # marginal distribution
    for iter in range(maxiter):
        optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        y_simu = soft_m_vec_partial(N, T, log_theta.exp(), gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
        y_simu = soft_get_SS(y_simu).reshape(-1, 52).T # the summary statistics
        y_simu = torch.cat([y_simu[i:(i+delay)].reshape(1, -1) for i in range(len(y_simu) - delay + 1)], dim=0) # delayed reconstruction
        # print(y_simu)
        u = gen_u(u_size, x_obs.shape[1]).to(device)
        x_obs_projected = ( u.unsqueeze(0).repeat(x_obs.shape[0], 1, 1) * x_obs.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)        
        y_simu_projected = ( u.unsqueeze(0).repeat(y_simu.shape[0], 1, 1) * y_simu.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)
        
        if metric == "squaredW2":
            if x_obs.shape[0] == y_simu.shape[0]:
                loss = samen_W2_squared_1d_vec(x_obs_projected, y_simu_projected).mean()
            else:
                loss = W2_squared_1d_vec(x_obs_projected, y_simu_projected).mean()
        if metric == "W1":
            if x_obs.shape[0] == y_simu.shape[0]:
                loss = samen_W1_1d_vec(x_obs_projected, y_simu_projected).mean()
            else:            
                loss = W1_1d_vec(x_obs_projected, y_simu_projected).mean()
        if metric == "naive": # bad
            loss = torch.linalg.norm(y_simu - x_obs)
        
        loss.backward()
        # theta.grad = torch.nan_to_num(theta.grad, nan=0.0)
        # print("gradient:", theta.grad)
        # print("theta:", theta.detach())
        optimizer.step()
        
        # no constraint 
        
        log_theta_path.append(log_theta.detach().clone())
        train_loss_path.append(loss.item())

    if plot == True:
        # Plot the training loss
        plt.figure(figsize=(8, 4))
        plt.plot(range(maxiter), train_loss_path, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Over Iterations')
        plt.show()
    return log_theta_path


def delay_SS_log_res_diffz(delay, metric, data_obs, simu_size, num_diffz, lr, maxiter):
    res_diffz = torch.zeros(num_diffz, 12, dtype = torch.float64)
    
    u_size = 100
    for i in range(num_diffz):
        # generate z
        z = gen_z(simu_size, T)
        
        # random initialization
        theta_init = 0.5 * torch.rand(12, dtype = torch.float64)
        log_theta_init = theta_init.log()
    
        # solve the SW objective by Adam
        log_theta_path = delay_SS_log_Adam_SW_fixz(delay, metric, data_obs, u_size, z, log_theta_init, lr, maxiter, plot = False)
    
        # record the solution
        res_diffz[i] = log_theta_path[-1]
    return res_diffz


def m_vec_partial32(N, T, beta, gamma, alpha, eta, F_assign, C_F, C_R, Z, NF, NR):
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
    X[:, 0] = Ind32(alpha - allU[:, 0]) # initialization
    Y[:, 0] = X[:, 0] # patients are screened when they enter
    for t in range(1, T):
        # First, update X
        D = allD[:, t] # discharge or not
        X[D == 1, t] = Ind32(alpha - allU[D == 1, t]) # for replaced patients
        norep_infec_id = torch.logical_and(D == 0, X[:, t-1] > 0.5) # people who are not replaced and have already been infected
        norep_sus_id = torch.logical_and(D == 0, X[:, t-1] < 0.5) # people who are not replaced and are not infected

        X[norep_infec_id, t] = 1 # X[norep_infec_id, t-1]
        lam = hazard32(beta, X[:, t-1], F_assign, C_F, C_R, N, NF, NR)
        lam = lam[norep_sus_id]
        X[norep_sus_id, t] = Ind32( (1 - (-lam).exp()) - allU[norep_sus_id, t] )

        # Next, get Y based on X
        Y[D == 1, t] = X[D == 1, t]
        id1 = torch.logical_and(X[:, t] > 0.5, Y[:, t-1] < 0.5)
        Y[torch.logical_and(D == 0, id1), t] = Ind32(eta - allV[torch.logical_and(D == 0, id1), t])
        Y[torch.logical_and(D == 0, ~id1), t] = Y[torch.logical_and(D == 0, ~id1), t - 1]
    return Y # , X

def hazard32(beta, X, F_assign, C_F, C_R, N, NF, NR):
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

def gen_z32(N, T):
    """
        generate the latent random variable
    """
    return [torch.bernoulli(gamma * torch.ones(N, T)).to(device), torch.rand(N, T).to(device), torch.rand(N, T).to(device)]

def Ind32(t):
    """
        Indicator function
    """
    return ( 1.0 * (t > 0) )