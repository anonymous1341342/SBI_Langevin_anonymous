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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 0.1
M = 10

# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0

lower = torch.zeros(M + 1).to(device)
lower[0] = a0
lower[1:] = a
upper = torch.zeros(M + 1).to(device)
upper[0] = b0
upper[1:] = b


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

def m_vec(theta, z):
    x = z[0]
    y = z[2] @ theta + z[1]
    return torch.cat((x.view(-1, 1), y.view(-1, 1)), dim=1)

def gen_z(sample_size):
    """
        # generate the latent random variable
        # z = (Uniform(0, 1), N(0, \sigma^2), Design(Uniform(0, 1))), where the first element is just (the observable) x, 
            and the second element is the noise when generating y, and the third element is the design matrix based on the
            first element x, we calculate this in the latent variable generation process to save computation, because for 
            different \theta in m_vec(), we use the same design matrix to generate y
    """
    x = torch.rand(sample_size).to(device)
    noise = sigma * torch.randn(sample_size).to(device)
    psi_x = get_psi(x, M)
    A = get_A(M)
    design_x = ( psi_x @ torch.linalg.inv(A) ).to(device)
    
    return [x, noise, design_x]

#############################
#        SW1 precond        #
#############################
def theta2alpha(theta):
    alpha = ( (theta - lower) / (upper - theta) ).log()
    return alpha

def alpha2theta(alpha):
    theta = (lower + upper * alpha.exp()) / (1 + alpha.exp())
    return theta

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

def gen_u(usize, udim):
    """
        Draw u from the Uniform distribution on the surface of the unit-L2ball
    """
    xi = torch.randn(usize, udim) # draw multivariate gaussian
    return xi / torch.linalg.norm(xi, dim = 1).view(-1, 1).repeat(1, udim)

def Adam_SW1_fixz_repar(x_obs, u_size, z, theta_init, lr, maxiter, scheduler_patience, early_stop_patience):
    """
    x_obs: observed x
    simu_size: the size of simulated data
    theta_init: initial value of theta
    lr: learning rate (step size) of gradient descent
    """

    x_obs = x_obs.to(device)
    theta = theta_init.to(device)
    alpha = theta2alpha(theta)
    alpha.requires_grad_(True)
    optimizer = optim.Adam([alpha], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, min_lr=1e-3
    )
    
    theta_path = [] # store the results
    alpha_path = []
    train_loss_path = []

    best_loss = float('inf')
    epochs_no_improve = 0

    # marginal distribution
    for it in range(maxiter):
        alpha_path.append(alpha.detach().clone().cpu())
        optimizer.zero_grad()
        theta = alpha2theta(alpha)
        y_simu = m_vec(theta, z)
        u = gen_u(u_size, x_obs.shape[1]).to(device)
        x_obs_projected = ( u.unsqueeze(0).repeat(x_obs.shape[0], 1, 1) * x_obs.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)
        y_simu_projected = ( u.unsqueeze(0).repeat(y_simu.shape[0], 1, 1) * y_simu.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)
        if x_obs.shape[0] == y_simu.shape[0]:
            SW1 = samen_W1_1d_vec(x_obs_projected, y_simu_projected).mean()
        SW1.backward()
        optimizer.step()
        
        theta_path.append(theta.detach().clone().cpu())
        train_loss_path.append(SW1.item())

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(SW1.item())
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Iteration {it}: Lr decreased to {new_lr:.2e}")

        # Early stopping check
        current_loss = SW1.item()
        if current_loss < best_loss - 1e-6:  # Add small delta to avoid floating point issues
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at iteration {it}: no improvement in last {early_stop_patience} steps.")
            break

    return theta_path, alpha_path, train_loss_path
