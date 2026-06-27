from utils_queuing_single import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd

# ===== Setting ===== #
obs_size = 500
theta_dim = 3
x_dim = 5

a1 = 0.0
b1 = 10.0

a2 = 0.0
b2 = 10.0

a3 = 0.01 #
b3 = 0.5


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


# =============== Load Configurations =============== #
# task_id = int(sys.argv[1])
# config = json.loads(sys.argv[2])

# NN structure of the original SingleModel
Orig_hidden_size = 64 # config["Orig_hidden_size"]
Orig_num_layers = 3 # config["Orig_num_layers"]

# NN structure of the Debias regression model
Deb_hidden_size = 64 #  config["Deb_hidden_size"]
Deb_num_layers = 3 # config["Deb_num_layers"]

extra_ref_size = int(1e4) # int(config["extra_ref_size"])
extra_obs_size = 500 # int(config["extra_obs_size"])

# Hyperparameters for Training the Deb Reg Model
batch_size = 10 # config["batch_size"]
num_epochs = 500 # config["num_epochs"]
early_stop_patience = 30 # config["early_stop_patience"]
learning_rate = 1e-3 # config["learning_rate"]
sched = True # config["sched"] == "True"
sched_patience = 10 # config["sched_patience"]


lam_curve = 1e-4 # config["lam_curve"]
pen_learning_rate = 3e-5 # config["pen_learning_rate"]
pen_num_epochs = 10 # config["pen_num_epochs"]


model_noise = 0.25 # config["model_noise"]
print(f"model_noise = {model_noise}")




def main(task_id):
    # Load the SingleModel_fisher
    SingleModel_fisher = ELU_single_LikeScoreMatchingNN(theta_dim, x_dim, Orig_hidden_size, Orig_num_layers).to(device)
    checkpoint = torch.load(f'model_single_fisher/mn{model_noise}/checkpoint_task{task_id}.pth', map_location=device)
    SingleModel_fisher.load_state_dict(checkpoint['model_state_dict'])
    bias_lastlayer = checkpoint['bias_lastlayer']

    with torch.no_grad(): 
        SingleModel_fisher.layers[-1].bias -= bias_lastlayer.to(device)


    # ============= ref_R ============== #
    path_theta = Path(f'ref_R/mn{model_noise}/theta_r0_extra_task{task_id}.npy')
    path_x = Path(f'ref_R/mn{model_noise}/x_r0_extra_task{task_id}.npy')
    path_theta.parent.mkdir(parents=True, exist_ok=True) # ensure parent folder exists


    theta_r0_extra = torch.from_numpy(np.load(path_theta))
    x_r0_extra = torch.from_numpy(np.load(path_x))
    print(f"dtype = {theta_r0_extra.dtype}, {x_r0_extra.dtype}")

    tr_size = int(0.9 * extra_ref_size)
    train_theta_r0_extra = theta_r0_extra[:tr_size]
    train_x_r0_extra = x_r0_extra[:tr_size]
    val_theta_r0_extra = theta_r0_extra[tr_size:]
    val_x_r0_extra = x_r0_extra[tr_size:]



    # =============== Use SingleModel_fisher to get the label =============== #
    # Create Label
    # use the extra data to estimate the mean score for each theta
    sample_size = train_theta_r0_extra.shape[0]
    bsize = max(10, int(sample_size / 100))
    SingleModel_fisher_est_M = torch.zeros(sample_size, 3)
    for i in range(int(sample_size / bsize)):
        theta = train_theta_r0_extra[(i*bsize):((i+1)*bsize)]
        data = train_x_r0_extra[(i*bsize):((i+1)*bsize)]
        SingleModel_fisher_est_M[(i*bsize):((i+1)*bsize)] = SingleModel_fisher.layers( torch.cat( (theta.to(device).repeat_interleave(extra_obs_size, dim = 0), data.to(device).reshape(-1, x_dim)), dim = 1 ) ).view(theta.shape[0], extra_obs_size, theta.shape[1]).sum(dim = 1).detach()

    mean_score = SingleModel_fisher_est_M / extra_obs_size

    print(f"mean_score.mean(dim = 0) = {mean_score.mean(dim = 0)}")
    print(f"mean_score.std(dim = 0) = {mean_score.std(dim = 0)}")


    # validation data
    sample_size = val_theta_r0_extra.shape[0]
    bsize = max(10, int(sample_size / 100))
    SingleModel_fisher_est_M = torch.zeros(sample_size, 3)
    for i in range(int(sample_size / bsize)):
        theta = val_theta_r0_extra[(i*bsize):((i+1)*bsize)]
        data = val_x_r0_extra[(i*bsize):((i+1)*bsize)]
        SingleModel_fisher_est_M[(i*bsize):((i+1)*bsize)] = SingleModel_fisher.layers( torch.cat( (theta.to(device).repeat_interleave(extra_obs_size, dim = 0), data.to(device).reshape(-1, x_dim)), dim = 1 ) ).view(theta.shape[0], extra_obs_size, theta.shape[1]).sum(dim = 1).detach()

    val_mean_score = SingleModel_fisher_est_M / extra_obs_size


    # prepare the weight
    train_weight = g(train_theta_r0_extra.cpu(), train_x_r0_extra.cpu())
    val_weight = g(val_theta_r0_extra.cpu(), val_x_r0_extra.cpu())

    # prepare dataloader
    train_dataset = TensorDataset(train_theta_r0_extra, mean_score, train_weight)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_theta_r0_extra, val_mean_score, val_weight)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    # ================== Train without curvature penalty ================== #
    DebReg_model = Deb_ELU(input_dim = theta_dim, output_dim = theta_dim, hidden_size = Deb_hidden_size, num_layers = Deb_num_layers).to(device)
    save_path = Path(f"DebRegModel_init/mn{model_noise}/checkpoint_task{task_id}.pth")
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)
            print(f"Scheduler is used, with patience {sched_patience}")
        
        train_weighted_DebReg(DebReg_model, optimizer, train_loader, val_loader, num_epochs, scheduler, early_stop_patience)
        torch.save(
            {'model_state_dict': DebReg_model.state_dict()}, 
            save_path
        )

        print("\n Bias before and after the debias regression (without penalty):")
        print(f"val_mean_score.abs().mean(dim = 0) = {val_mean_score.cpu().abs().mean(dim = 0)}")
        print("\n")
        print(f"( ( val_mean_score - DebReg_model(val_theta_r0_extra) ).abs() ).mean(dim = 0) = {( ( val_mean_score.to(device) - DebReg_model(val_theta_r0_extra.to(device)).detach() ).abs() ).mean(dim = 0)}")


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


    save_path_fisher = Path(f"DebRegModel_fisher/mn{model_noise}/checkpoint_task{task_id}.pth")
    save_path_fisher.parent.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save({'model_state_dict': DebReg_model.state_dict()}, save_path_fisher)


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)

