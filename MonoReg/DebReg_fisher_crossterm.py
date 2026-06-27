from utils_monoBP_single import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import json
import sys


def check_loss(model, val_loader, lam_curve):
    loss_fn = nn.MSELoss()
    model.eval()
    val_total_obj = 0.0
    val_total_reg_loss = 0.0
    val_total_curve_loss = 0.0
    val_total_scale_ggT = 0.0
    # with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        pred_predT = pred.unsqueeze(2) @ pred.unsqueeze(1)
        jacob = Deb_curve(model, xb)
        loss_reg = loss_fn(pred, yb)
        loss_curve = loss_fn( pred_predT, jacob + torch.einsum('bi,bj->bij', yb, pred) + torch.einsum('bi,bj->bij', pred, yb) )
        loss = loss_reg + lam_curve * loss_curve
        scale_ggT = loss_fn(pred_predT, torch.zeros_like(pred_predT))
        
        val_total_obj += loss.item() * xb.size(0)
        val_total_reg_loss += loss_reg.item() * xb.size(0)
        val_total_curve_loss += loss_curve.item() * xb.size(0)
        val_total_scale_ggT += scale_ggT.item() * xb.size(0)
        
    avg_val_obj = val_total_obj / len(val_loader.dataset)
    avg_val_reg_loss = val_total_reg_loss / len(val_loader.dataset)
    avg_val_curve_loss = val_total_curve_loss / len(val_loader.dataset)
    avg_val_total_scale_ggT = val_total_scale_ggT / len(val_loader.dataset)

    print(f"Val Loss (Total, Reg, Curve): ({avg_val_obj:.6f}, {avg_val_reg_loss:.6f},{avg_val_curve_loss:.6f})")
    print(f"Scale of ggT = {avg_val_total_scale_ggT:.6f}")
    return avg_val_reg_loss


# =============== Load Configurations =============== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.version.cuda) 
print(torch.cuda.is_available()) 


theta_dim= M + 1
x_dim = 2
obs_size = 1000

# NN structure of the original SingleModel
Orig_hidden_size = 64 # config["Orig_hidden_size"]
Orig_num_layers = 3 # config["Orig_num_layers"]

# NN structure of the Debias regression model
Deb_hidden_size = 64 # config["Deb_hidden_size"]
Deb_num_layers = 3 # config["Deb_num_layers"]

extra_sample_size = 50000 # int(config["extra_sample_size"])
extra_obs_size = 1000 # int(config["extra_obs_size"])

batch_size = 256 # config["batch_size"]
num_epochs = 300 # config["num_epochs"]
early_stop_patience = 30 # config["early_stop_patience"]
learning_rate = 1e-3 # config["learning_rate"]
sched = True # config["sched"] == "True"
sched_patience = 10 # int(config["sched_patience"])

lam_curve = 1e-3 # config["lam_curve"]
pen_learning_rate = 1e-5 # config["pen_learning_rate"]
pen_num_epochs = 100 # config["pen_num_epochs"]



def main(task_id):
    # =============== Debias Regression Model and Training Functions =============== #
    # Load the SingleModel_fisher
    SingleModel_fisher = single_ELU_LikeScoreMatchingNN(theta_dim, x_dim, obs_size, Orig_hidden_size, Orig_num_layers).to(device)
    checkpoint = torch.load(f'model_single_fisher/checkpoint_task{task_id}_trainsize{int(1e6)}.pth', map_location=device)
    SingleModel_fisher.load_state_dict(checkpoint['model_state_dict'])
    bias_lastlayer = checkpoint['bias_lastlayer']

    with torch.no_grad(): 
        SingleModel_fisher.layers[-1].bias -= bias_lastlayer.to(device)


    # =============== Generate Data For Training the Deb Reg Model =============== #
    # Load preconditioned samples
    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()

    ### generate training data
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



    # Load training samples for Deb Reg
    theta_r0_extra = torch.from_numpy(np.load(f'extra_data/theta_r0_extra_task{task_id}.npy'))[:extra_sample_size]
    data_r0_extra = torch.from_numpy(np.load(f'extra_data/data_r0_extra_task{task_id}.npy'))[:extra_sample_size]

    val_theta_r0_extra, val_data_r0_extra = gen_ref(mean_theta, std_theta, lower, upper, extra_obs_size, extra_sample_size + 100)

    bad_mask = torch.isinf(val_theta_r0_extra).any(dim=1)
    val_theta_r0_extra = val_theta_r0_extra[~bad_mask]
    val_data_r0_extra = val_data_r0_extra[~bad_mask]
    val_theta_r0_extra = val_theta_r0_extra[:extra_sample_size]
    val_data_r0_extra = val_data_r0_extra[:extra_sample_size]

    # =============== Use SingleModel_fisher to get the label =============== #
    # Create Label
    # use the extra data to estimate the mean score for each theta
    sample_size = theta_r0_extra.shape[0]
    x_dim = 2
    bsize = max(10, int(sample_size / 100))
    SingleModel_fisher_est_M = torch.zeros(sample_size, 11)
    for i in range(int(sample_size / bsize)):
        theta = theta_r0_extra[(i*bsize):((i+1)*bsize)]
        data = data_r0_extra[(i*bsize):((i+1)*bsize)]
        SingleModel_fisher_est_M[(i*bsize):((i+1)*bsize)] = SingleModel_fisher.layers( torch.cat( (theta.to(device).repeat_interleave(extra_obs_size, dim = 0), data.to(device).reshape(-1, x_dim)), dim = 1 ) ).view(theta.shape[0], extra_obs_size, theta.shape[1]).sum(dim = 1).detach()

    mean_score = SingleModel_fisher_est_M / extra_obs_size


    # validation data
    sample_size = val_theta_r0_extra.shape[0]
    x_dim = 2
    bsize = max(10, int(sample_size / 100))
    SingleModel_fisher_est_M = torch.zeros(sample_size, 11)
    for i in range(int(sample_size / bsize)):
        theta = val_theta_r0_extra[(i*bsize):((i+1)*bsize)]
        data = val_data_r0_extra[(i*bsize):((i+1)*bsize)]
        SingleModel_fisher_est_M[(i*bsize):((i+1)*bsize)] = SingleModel_fisher.layers( torch.cat( (theta.to(device).repeat_interleave(extra_obs_size, dim = 0), data.to(device).reshape(-1, x_dim)), dim = 1 ) ).view(theta.shape[0], extra_obs_size, theta.shape[1]).sum(dim = 1).detach()

    val_mean_score = SingleModel_fisher_est_M / extra_obs_size

    # prepare dataloader
    train_dataset = TensorDataset(theta_r0_extra, mean_score)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_theta_r0_extra, val_mean_score)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # train the model without curvature penalty first
    DebReg_model = Deb_ELU(input_dim = M+1, output_dim = M+1, hidden_size = Deb_hidden_size, num_layers = Deb_num_layers)
    optimizer = optim.Adam(DebReg_model.parameters(), lr=learning_rate)
    scheduler = None
    if sched:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)
        print(f"Scheduler is used, with patience {sched_patience}")

    train_DebReg(DebReg_model, optimizer, train_loader, val_loader, num_epochs, scheduler, early_stop_patience)


    # print the loss information after the initialization
    print("Loss information after initialization without penalty:")
    best_val_reg_loss = check_loss(DebReg_model, val_loader, lam_curve)
    print("\n")


    # Add the curvature penalty and continue to train
    optimizer = optim.Adam(DebReg_model.parameters(), lr=pen_learning_rate)
    train_DebReg_fisher_crossterm(DebReg_model, optimizer, train_loader, val_loader, pen_num_epochs, lam_curve, best_val_reg_loss)

    torch.save(DebReg_model, f'DebRegModel_fisher_crossterm/model_task{task_id}.pth')


if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)


