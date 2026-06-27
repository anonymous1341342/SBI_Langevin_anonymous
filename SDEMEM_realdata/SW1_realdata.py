from utils_SDEMEM_realdata import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd

print(f"Using device: {device}")

# ========== load observed data
df = pd.read_excel("realdata/20160427_mean_eGFP.xlsx", header=None)
x_obs = torch.tensor(df.to_numpy(), dtype=torch.float32)[:, 1:].T.log().to(device)

obs_size = 40 # 200
x_obs = x_obs[:obs_size]
print(f"x_obs shape: {x_obs.shape}")

# just for initialization
# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
init_mean_tmp = torch.tensor([5, 1, 3, -1, -1, -5, 0.5, 0, -0.8, -0.8, -0.8, -0.8], dtype = torch.float32)
init_std_tmp = torch.tensor([1, 1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype = torch.float32)

T = 30
theta_dim = 12
x_dim = 180 



def gen_u(usize, udim, device, dtype):
    """
        Draw u from the Uniform distribution on the surface of the unit-L2ball
    """
    xi = torch.randn(usize, udim, device = device, dtype = dtype) # draw multivariate gaussian
    return xi / torch.linalg.norm(xi, dim = 1).view(-1, 1).repeat(1, udim)

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

def Adam_SW1_fixz(x_obs, T, u_size, seed, theta_init, lr, maxiter, scheduler_patience, early_stop_patience, opt = 'Adam', plot = True, mute = False):
    # Input:
    # x_obs: tensor, observed x, [N, x_dim]
    # T: scalar, total time
    # u_size: number of projections used in calculating the SW1 distance
    # seed: it plays the role of z, as it determines the latent random variables
    # theta_init: initial value of theta, [theta_dim, ] or [1, theta_dim]
    # lr: learning rate (step size) of gradient descent
    
    device = theta_init.device
    dtype = theta_init.dtype

    x_obs = x_obs.to(device)
    theta = theta_init.detach().clone().to(device).reshape(1, -1)
    theta.requires_grad_(True)

    if opt == 'Adam':
        print('Using Adam optimizer')
        optimizer = optim.Adam([theta], lr=lr)

    if opt == 'SGD':
        print('Using SGD optimizer')
        optimizer = optim.SGD([theta], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, min_lr=1e-2
    ) # factor=0.5
    
    theta_path = [] # store the results
    train_loss_path = []

    best_loss = float('inf')
    epochs_no_improve = 0

    # marginal distribution
    for iter in range(maxiter):
        t1 = time.time()
        optimizer.zero_grad()
        y_simu = gen_x_given_theta(theta.repeat(x_obs.shape[0], 1), T = T, epis = 0.01, len_interpolate = 1/6, seed = seed, mute = True, use_soft_Ind = True)
        u = gen_u(u_size, x_obs.shape[1], device, dtype)
        x_obs_projected = ( u.unsqueeze(0).repeat(x_obs.shape[0], 1, 1) * x_obs.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)
        y_simu_projected = ( u.unsqueeze(0).repeat(y_simu.shape[0], 1, 1) * y_simu.unsqueeze(1).repeat(1, u_size, 1) ).sum(dim = 2)
        if x_obs.shape[0] == y_simu.shape[0]:
            SW1 = samen_W1_1d_vec(x_obs_projected, y_simu_projected).mean()
        
        theta_path.append(theta.detach().clone())
        train_loss_path.append(SW1.item())
        
        SW1.backward()
        
        with torch.no_grad():
            bad_grad = False
            if theta.grad is not None:
                bad_grad = (not torch.isfinite(theta.grad).all().item())
                theta.grad = torch.nan_to_num(theta.grad, nan=0.0, posinf=0.0, neginf=0.0)
                g = theta.grad.detach().view(-1)
                g_abs = g.abs()

        if bad_grad:
            print(f"[iter {iter}] non-finite or nan grad detected, stopping the algorithm.")
            return None, None

        
        # print(f"grad = {theta.grad}")
        t2 = time.time()
        if not mute:
            print(f"Iteration {iter+1}/{maxiter} | loss = {SW1.item():.3f} | time = {t2-t1:.2f} seconds")

        optimizer.step()

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(SW1.item())
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr and (not mute):
            print(f"Iteration {iter}: Lr decreased to {new_lr:.2e}")

        # Early stopping check
        current_loss = SW1.item()
        if current_loss < best_loss - 1e-6:  
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at iteration {iter}: no improvement in last {early_stop_patience} steps.")
            break

    
    if plot == True:
        # Plot the training loss
        plt.figure(figsize=(8, 4))
        plt.plot(range(iter + 1), train_loss_path, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Over Iterations')
        plt.show()
    return theta_path, train_loss_path


def res_diffz(x_obs, num_diffz):
    theta_res_diffz = torch.zeros(num_diffz, theta_dim)
    final_loss = torch.zeros(num_diffz)
    
    u_size = 100
    lr = 1e-1
    maxiter = 120 # 200 

    scheduler_patience = 10
    early_stop_patience = 20
    opt = 'Adam'
    
    for i in range(num_diffz):
        t1 = time.time()
        # print(f"\nRunning {i+1}/{num_diffz}")
        
        # random initialization
        theta_init = init_mean_tmp + init_std_tmp * torch.randn(1, theta_dim)
        theta_init = theta_init.to(torch.float64)
    
        # solve the SW objective by Adam
        seed = i + 2000
        theta_path, train_loss_path = Adam_SW1_fixz(x_obs, T, u_size, seed, theta_init, lr, maxiter, scheduler_patience, early_stop_patience, opt, plot = False, mute = True)

        if theta_path is None or train_loss_path is None:
            print(f"Failed to converge for diffz {i+1}.")
            theta_res_diffz[i] = torch.full((theta_dim,), float('nan'))
            final_loss[i] = float('nan')
            continue

        best_idx = int(np.argmin(train_loss_path))
        theta_solu = theta_path[best_idx].clone().ravel()   

        # record the solution
        theta_res_diffz[i] = theta_solu
        final_loss[i] = train_loss_path[best_idx]

        t2 = time.time()
        print(f"Finished {i+1}/{num_diffz} in {t2-t1:.2f} seconds. Best loss = {train_loss_path[best_idx]:.3f}")
    return theta_res_diffz, final_loss


def main():
    # ===== SW1 Localization ===== #
    theta_SW1, final_loss = res_diffz(x_obs, num_diffz = 110) # add 10 more runs, because some runs may fail, due to the overflow in the soft_Ind function, if the constant is too large or the initialization is bad.

    save_dir = Path("res_SW1")
    save_dir.mkdir(parents=True, exist_ok=True)

    # assert torch.isfinite(theta_SW1).all(), f"Non-finite values found in theta_SW1"
    np.save(save_dir / f"theta_SW1.npy", theta_SW1.detach().cpu().numpy())
    np.save(save_dir / f"final_loss.npy", final_loss.detach().cpu().numpy())


if __name__ == "__main__":
    main()