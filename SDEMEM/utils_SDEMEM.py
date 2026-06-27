import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import math
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import time
from pathlib import Path
import sys
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def Ind(t):
    """
        Indicator function
    """
    return 1.0 * (t > 0)

# def soft_Ind(t):
#     """
#         Indicator function
#     """
#     return 1 / ( 1 + (-15.0 * t).exp() ) # -15


# smooth version of indicator function
def soft_Ind(t):
    return torch.sigmoid(1000.0 * t)


def gen_x_given_theta(theta, T = 30, epis = 0.01, len_interpolate = 1/6, seed = None, mute = False, use_soft_Ind = False):
    """
        generate x given theta, vectorized version
    """
    # Input:
    # theta: [N, theta_dim], where the columns of theta are respectively: 
    # [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
    # T: scalar, total length of the time series
    # epis: step size in the Euler discretization
    # len_interpolate: record the obsesrvation every {interpolate_length} time

    # Output:
    # y: [N, T/len_interpolate]. Note: we first generate two tensors of dimension [N, T/len_interpolate], each recording the 'm' response and the 'p' response

    # Note: the tau here is parameterized differently from the paper. Instead of modelling log_c ~ N(mu, \tau^{-1}), here we model log_c ~ N(mu, \tau^2) for simplicity,
    # then we will assign normal proposal/prior to all our parameters

    # We let dm and dp to be be 0 before t0. This would keep m and p to be m0 and 0 before t0, this p matches the model, but this m before t0 is wrong.
    # However, this still gives us the correct final output, since the output only depend on p. If we want to also get correct m, just set dm = 0 before t0 in the end.


    if seed is not None:
        set_seed(seed)

    log_m0 = theta[:, 0]
    log_scale = theta[:, 1]
    log_offset = theta[:, 2]
    log_sigma = theta[:, 3]
    
    mu = theta[:, 4:8]
    log_tau = theta[:, 8:]
    
    N = theta.shape[0]
    L = int(T/len_interpolate)
    device = theta.device
    dtype = theta.dtype
    
    res_m = torch.zeros(N, L, device=device, dtype=dtype)
    res_p = torch.zeros(N, L, device=device, dtype=dtype)

    t = 0.0
    t_copy = 0.0 # to decide whether to record
    idx_L = 0 # column index of the res_m or res_p for recording the values
    m_prev = log_m0.exp() # [N, ]
    p_prev = torch.zeros(N, device=device, dtype=dtype)

    # Draw c^{(i)}: log_c ~ N(mu, tau^2)
    # log_c = mu + log_tau.exp() * torch.randn(mu.shape, device = device) # [N, 3], log_c.exp() = [delta, gamma, k]
    c = ( mu + log_tau.exp() * torch.randn_like(mu) ).exp() # [N, 4], c = [delta, gamma, k, t0]
    
    delta = c[:, 0]
    gamma = c[:, 1]
    k = c[:, 2]
    t0 = c[:, 3]

    num_record = 0
    while t <= T:
        t += epis
        t_copy += epis
        # m_curr = m_prev + epis * (-delta * m_prev) + (delta * m_prev).sqrt() * math.sqrt(epis) * torch.randn_like(m_prev)
        # p_curr = p_prev + epis * (k * m_prev - gamma * p_prev) + (k * m_prev + gamma * p_prev).sqrt() * math.sqrt(epis) * torch.randn_like(p_prev)

        eps_var = 1e-20
        dm = epis * (-delta * m_prev) + torch.clamp(delta * m_prev, min=eps_var).sqrt() * math.sqrt(epis) * torch.randn_like(m_prev)
        dp = epis * (k * m_prev - gamma * p_prev) + torch.clamp(k * m_prev + gamma * p_prev, min=eps_var).sqrt() * math.sqrt(epis) * torch.randn_like(p_prev)

        # the dynamic is muted before t0
        if use_soft_Ind:
            m_curr = m_prev + dm * soft_Ind(t-t0)
            p_curr = p_prev + dp * soft_Ind(t-t0)
        else:
            m_curr = m_prev + dm * Ind(t-t0)
            p_curr = p_prev + dp * Ind(t-t0)

        # truncate to be non-negative
        m_curr = torch.relu(m_curr)
        p_curr = torch.relu(p_curr)


        m_prev = m_curr
        p_prev = p_curr

        # if t % len_interpolate == 0:
        # if (math.isclose(t % len_interpolate, 0.0, abs_tol=1e-6) or math.isclose(t % len_interpolate, len_interpolate, abs_tol=1e-6)): # record
        if t_copy >= len_interpolate: # record
            t_copy -= len_interpolate
            num_record += 1
            # print(f"record, t = {t}")
            res_m[:, idx_L] = m_curr
            res_p[:, idx_L] = p_curr
            idx_L += 1
    if not mute:
        print(f"total number of records = {num_record}")

    # get the observed y from p
    res_y = ( log_scale.exp().reshape(-1, 1) * res_p + log_offset.exp().reshape(-1, 1) ).log() + log_sigma.exp().reshape(-1, 1) * torch.randn_like(res_p)
    
    return res_y 