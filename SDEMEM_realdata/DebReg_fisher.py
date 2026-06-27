from utils_sm import *
from utils_SDEMEM_realdata import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path

# ===== Setting for real data ===== #
T = 30
theta_dim = 12
x_dim = 180 


# =============== Load Configurations =============== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# NN structure of the original SingleModel
Orig_hidden_size = 128 # config["Orig_hidden_size"]
Orig_num_layers = 3 # config["Orig_num_layers"]

# NN structure of the Debias regression model
Deb_hidden_size = 128 # config["Deb_hidden_size"]
Deb_num_layers = 3 # config["Deb_num_layers"]

extra_ref_size = int(1e6) # int(config["extra_ref_size"])
extra_obs_size = 40 # int(config["extra_obs_size"])

batch_size = 64 # config["batch_size"]
num_epochs = 300 # config["num_epochs"]
early_stop_patience = 20 # config["early_stop_patience"]
learning_rate = 1e-4 # config["learning_rate"]
sched = True


lam_curve = 1e-2 # config["lam_curve"]
pen_learning_rate = 1e-4 # config["pen_learning_rate"]
pen_num_epochs = 1 # config["pen_num_epochs"]


sm_rd = 1


def main(task_id):
    def check_loss(model, val_loader, lam_curve):
        loss_fn = nn.MSELoss()
        model.eval()
        val_total_obj = 0.0
        val_total_reg_loss = 0.0
        val_total_curve_loss = 0.0
        val_total_scale_ggT = 0.0
        val_total_scale_sgT = 0.0
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
            scale_ggT = loss_fn(pred_predT * weight_tensor, torch.zeros_like(pred_predT))
            scale_sgT = loss_fn(torch.einsum('bi,bj->bij', yb, pred) * weight_tensor, torch.zeros_like(pred_predT))
            
            val_total_obj += loss.item() * xb.size(0)
            val_total_reg_loss += loss_reg.item() * xb.size(0)
            val_total_curve_loss += loss_curve.item() * xb.size(0)
            val_total_scale_ggT += scale_ggT.item() * xb.size(0)
            val_total_scale_sgT += scale_sgT.item() * xb.size(0)
            
        avg_val_obj = val_total_obj / len(val_loader.dataset)
        avg_val_reg_loss = val_total_reg_loss / len(val_loader.dataset)
        avg_val_curve_loss = val_total_curve_loss / len(val_loader.dataset)
        avg_val_total_scale_ggT = val_total_scale_ggT / len(val_loader.dataset)
        avg_val_total_scale_sgT = val_total_scale_sgT / len(val_loader.dataset)

        print(f"Val Loss (Total, Reg, Curve): ({avg_val_obj:.6f}, {avg_val_reg_loss:.6f},{avg_val_curve_loss:.6f})")
        print(f"Scale of ggT = {avg_val_total_scale_ggT:.6f}")
        print(f"Scale of sgT = {avg_val_total_scale_sgT:.6f}")
        return avg_val_reg_loss



    # =============== Debias Regression Model and Training Functions =============== #
    # Load the FisherModel
    checkpoint_path = f"scaled_fishermodel_weighted/sm_round{sm_rd}/checkpoint_task{task_id}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    SingleModel_init = ELU_single_LikeScoreMatchingNN_sparse(theta_dim, x_dim, Orig_hidden_size, Orig_num_layers).to(device)
    SingleModel_init.load_state_dict(checkpoint['model_state_dict'])
    bias_lastlayer = checkpoint['bias_lastlayer']

    with torch.no_grad(): 
        SingleModel_init.net[-1].bias -= bias_lastlayer.to(device)


    path_val_loss_all_dim = checkpoint['path_val_loss_all_dim']
    path_val_loss_all_dim = np.stack(path_val_loss_all_dim, axis = 0) # shape (num_epochs, theta_dim)
    scale_score = torch.tensor( -path_val_loss_all_dim.min(axis=0), dtype = torch.float32)
    print(f"Using scale_score = {scale_score}")

    mean_x = checkpoint['mean_x'].cpu().repeat(1, extra_obs_size)
    std_x = checkpoint['std_x'].cpu().repeat(1, extra_obs_size)


    # CHANGED THE WEIGHT HERE
    def g(theta, x):
        return torch.ones_like(theta) / scale_score.to(theta.device)

    def g1(theta, x):
        return torch.zeros_like(theta)


    # ==== Load SW result ==== #
    theta_SW1 = np.load("res_SW1/theta_SW1.npy")
    loss_SW1 = np.load("res_SW1/final_loss.npy")

    nan_idx = np.isnan(theta_SW1).any(axis=1)
    theta_SW1 = theta_SW1[~nan_idx]
    loss_SW1 = loss_SW1[~nan_idx]

    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)[:100]
    print(f"theta_SW1.shape = {theta_SW1.shape}")


    prop_mean = theta_SW1.mean(dim = 0, keepdims = True)
    prop_std = theta_SW1.std(dim = 0, keepdims = True) 

    # inflate the proposal std
    prop_std *= 2

    # the previous prop_std is too small for these two dimensions
    prop_std[0, 0] *= 3
    prop_std[0, 6] *= 3


    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")


    # =============== Generate Data For Training the Deb Reg Model =============== #
    path_theta = Path(f'ref_R/sm_round{sm_rd}/theta_r0_extra_task{task_id}.npy')
    path_x = Path(f'ref_R/sm_round{sm_rd}/x_r0_extra_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists
    if not (path_theta.exists() and path_x.exists()):
        theta_r0_extra = torch.empty(extra_ref_size, theta_dim)
        x_r0_extra = torch.empty(extra_ref_size, extra_obs_size * x_dim)

        step = max(1, int(extra_ref_size / 4)) # max(1, int(extra_ref_size / 10))
        kept = 0

        for start in range(0, extra_ref_size, step):
            end = min(start + step, extra_ref_size)
            current_n = end - start

            theta_part = prop_mean + prop_std * torch.randn(current_n, theta_dim)
            x_part = torch.empty(current_n, extra_obs_size * x_dim)

            for i in range(extra_obs_size):
                x_part[:, i * x_dim:(i + 1) * x_dim] = gen_x_given_theta(theta_part.to(device), T=T).cpu()

            valid_mask = torch.isfinite(x_part).all(dim=1)
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                continue

            theta_r0_extra[kept:kept + num_valid].copy_(theta_part[valid_mask])
            x_r0_extra[kept:kept + num_valid].copy_(x_part[valid_mask])
            kept += num_valid

        theta_r0_extra = theta_r0_extra[:kept]
        x_r0_extra = x_r0_extra[:kept]

        print(f"generated reference table with shape theta: {theta_r0_extra.shape}, x: {x_r0_extra.shape}")

        np.save(path_theta, theta_r0_extra.numpy())
        np.save(path_x, x_r0_extra.numpy())
    else:
        theta_r0_extra = torch.from_numpy(np.load(path_theta))
        x_r0_extra = torch.from_numpy(np.load(path_x))
        print(f"extra data dtype = {theta_r0_extra.dtype}, {x_r0_extra.dtype}")

    # standardize theta
    theta_r0_extra = (theta_r0_extra - prop_mean) / prop_std
    x_r0_extra -= mean_x
    x_r0_extra /= std_x

    # split train and validation
    N_tr = int(0.9 * extra_ref_size)
    train_theta_extra = theta_r0_extra[:N_tr]
    train_x_extra = x_r0_extra[:N_tr]

    val_theta_extra = theta_r0_extra[N_tr:]
    val_x_extra = x_r0_extra[N_tr:]

    print(f"Number of training data = {train_theta_extra.shape[0]}")
    print(f"Number of validation data = {val_theta_extra.shape[0]}")

    # =============== Use SingleModel_init to get the label =============== #
    # Create Label
    # use the extra data to estimate the mean score for each theta
    sample_size = train_theta_extra.shape[0]
    bsize = max(100, int(sample_size / 1000))
    SingleModel_init_est_M = torch.zeros(sample_size, theta_dim)

    for start in range(0, sample_size, bsize):
        end = min(start + bsize, sample_size)
        theta = train_theta_extra[start:end]
        data = train_x_extra[start:end]
        SingleModel_init_est_M[start:end] = SingleModel_init.cal_penalty(theta.to(device), data.to(device)).sum(dim = 1).detach()

    mean_score = SingleModel_init_est_M / extra_obs_size

    print(f"mean_score.mean(dim = 0) = {mean_score.mean(dim = 0)}")
    print(f"mean_score.std(dim = 0) = {mean_score.std(dim = 0)}")


    # validation data
    sample_size = val_theta_extra.shape[0]
    bsize = max(100, int(sample_size / 1000))
    SingleModel_init_est_M = torch.zeros(sample_size, theta_dim)

    for start in range(0, sample_size, bsize):
        end = min(start + bsize, sample_size)
        theta = val_theta_extra[start:end]
        data = val_x_extra[start:end]
        SingleModel_init_est_M[start:end] = SingleModel_init.cal_penalty(theta.to(device), data.to(device)).sum(dim = 1).detach()

    val_mean_score = SingleModel_init_est_M / extra_obs_size


    # prepare the weight
    train_weight = g(train_theta_extra.cpu(), train_x_extra.cpu())
    val_weight = g(val_theta_extra.cpu(), val_x_extra.cpu())

    # prepare dataloader
    train_dataset = TensorDataset(train_theta_extra, mean_score, train_weight)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_theta_extra, val_mean_score, val_weight)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # ================== Train without curvature penalty ================== #
    DebReg_model = Deb_ELU_sparse(input_dim = theta_dim, output_dim = theta_dim, hidden_size = Deb_hidden_size, num_layers = Deb_num_layers).to(device)
    save_path = Path(f"DebRegModel_init/sm_round{sm_rd}/checkpoint_task{task_id}.pth")
    print("Using sparse NN for Debias Regression")


    save_path.parent.mkdir(parents=True, exist_ok=True)  # create folder if missing
    if save_path.exists():
        # load DebReg_model states
        checkpoint = torch.load(save_path, map_location = device)
        DebReg_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # train the model
        print("Train the Reg model without penalty")
        optimizer = optim.Adam(DebReg_model.parameters(), lr=learning_rate)
        scheduler = None
        if sched:
            sched_patience = 4 # int(config["sched_patience"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)
            print(f"Scheduler is used, with patience {sched_patience}")
        
        train_weighted_DebReg(DebReg_model, optimizer, train_loader, val_loader, num_epochs, scheduler, early_stop_patience)
        torch.save(
            {'model_state_dict': DebReg_model.state_dict()}, 
            save_path
        )

        print("\n Bias before and after the debias regression (without penalty):")
        print(f"val_mean_score.abs().mean(dim = 0) = {val_mean_score.cpu().abs().mean(dim = 0)}")
        print(f"(val_mean_score * val_weight**(1/2)).abs().mean(dim = 0) = {(val_mean_score.cpu() * val_weight.cpu()**(1/2)).abs().mean(dim = 0)}")
        print("\n")
        print(f"( ( val_mean_score - DebReg_model(val_theta_extra) ).abs() ).mean(dim = 0) = {( ( val_mean_score.to(device) - DebReg_model(val_theta_extra.to(device)).detach() ).abs() ).mean(dim = 0)}")
        print(f"( ( (val_mean_score - DebReg_model(val_theta_extra)) * val_weight**(1/2) ).abs() ).mean(dim = 0) = {( ( (val_mean_score.to(device) - DebReg_model(val_theta_extra.to(device)).detach()) * val_weight.to(device)**(1/2) ).abs() ).mean(dim = 0)}")

    # # ================== Add the curvature penalty to continue training ================== #
    print("Loss information after initialization without penalty:")
    best_val_reg_loss = check_loss(DebReg_model, val_loader, lam_curve)
    print("\n")
    print("\n")
    print("Now train with the curvature penalty")

    optimizer = optim.Adam(DebReg_model.parameters(), lr=pen_learning_rate)
    train_weighted_DebReg_fisher_crossterm(DebReg_model, optimizer, train_loader, val_loader, pen_num_epochs, lam_curve, best_val_reg_loss)

    # print the loss information after penalization
    print("Loss information after training with penalty:")
    best_val_reg_loss = check_loss(DebReg_model, val_loader, lam_curve)
    print("\n")


    save_path_fisher = Path(f"DebRegModel_fisher/sm_round{sm_rd}/checkpoint_task{task_id}.pth")
    save_path_fisher.parent.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save({'model_state_dict': DebReg_model.state_dict()}, save_path_fisher)



if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)