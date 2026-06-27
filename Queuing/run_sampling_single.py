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

model_noise = 0.25


def proj(theta):
    theta_proj = torch.zeros_like(theta)
    theta_proj[:, 0] = torch.clamp(theta[:, 0], a1, b1)
    theta_proj[:, 1] = torch.clamp(theta[:, 1], a2, b2)
    theta_proj[:, 2] = torch.clamp(theta[:, 2], a3, b3)
    return theta_proj


def proj_draw_post_vec(NScore_DebReged, x_obs, theta_init, epis, S):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_chain = theta_init.shape[0]
    
    theta0 = theta_init.to(device) # initial value of theta
    x_obs = x_obs.to(device).view(1, -1)
    epis = epis.to(device).view(1, -1)
    samples = torch.zeros(n_chain, S, theta_dim)

    for i in range(S):
        like_score_hat = NScore_DebReged(theta0, x_obs.repeat(n_chain, 1)).detach().to(device) # [n_chain, theta_dim]
        prior_score = 0.0
        theta1 = theta0 + epis * (like_score_hat + prior_score) + torch.sqrt(2.0 * epis) * torch.randn(theta0.shape).to(device) # draw a new sample
        theta1 = proj(theta1)
        theta0 = theta1 
        
        samples[:, i, :] = theta1.cpu().clone()
    return samples


def main():
    # ensure the folder "res_single_model" exists
    Path("res_single_model").mkdir(exist_ok=True)

    for obs_id in range(100):
        task_id= obs_id // 10
        
        # ======== Load the SingleModel ======== #
        checkpoint = torch.load(f'model_single_fisher/mn{model_noise}/checkpoint_task{task_id}.pth', map_location=device)

        model = ELU_single_LikeScoreMatchingNN(theta_dim, x_dim, 64, 3).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        bias_lastlayer = checkpoint['bias_lastlayer']

        with torch.no_grad(): 
            model.layers[-1].bias -= bias_lastlayer.to(device)


        # ======== Load the debreg model ======= #
        save_path = Path(f"DebRegModel_fisher/mn{model_noise}/checkpoint_task{task_id}.pth")
        DebReg_model = Deb_ELU(input_dim = theta_dim, output_dim = theta_dim, hidden_size = 64, num_layers = 3).to(device)

        checkpoint = torch.load(save_path, map_location = device)
        DebReg_model.load_state_dict(checkpoint['model_state_dict'])


        def NScore_DebReged(theta, x): # the input x is "N-data"
            return model.cal_penalty(theta, x).sum(dim = 1) - DebReg_model(theta) * obs_size


        x_obs = pd.read_csv(f"data_obs/x_obs_task{obs_id}.csv")
        x_obs = torch.tensor(x_obs.values, dtype = torch.float32).contiguous()


        epis = torch.tensor([0.01, 0.1, 0.001], dtype = torch.float32) / obs_size

        S = 2000
        n_chain = 10
        theta_init = ( torch.tensor([a1 + b1, a2 + b2, a3 + b3]) / 2 ).repeat(n_chain, 1)


        all_samples = []
        for _ in range(3):
            samples_nchain = proj_draw_post_vec(NScore_DebReged, x_obs + model_noise * torch.randn_like(x_obs), theta_init, epis, S)
            all_samples.append(samples_nchain.cpu().clone())
        np.save(f"res_single_model/samples_obs{obs_id}.npy", torch.cat(all_samples, dim = 0).cpu().numpy())


if __name__ == "__main__":
    main()