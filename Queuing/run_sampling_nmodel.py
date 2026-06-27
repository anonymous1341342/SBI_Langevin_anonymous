from utils_queuing_nmodel import *
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


def proj_vec(theta, x_obs):
    upper = torch.tensor([np.minimum(x_obs.min().item(), b1), b2, b3], dtype = torch.float32).to(device).repeat(theta.shape[0], 1)
    lower = torch.tensor([a1, a2, a3], dtype = torch.float32).to(device).repeat(theta.shape[0], 1)
    theta = torch.min(theta, upper)
    theta = torch.max(theta, lower)
    return theta

def proj_draw_post_vec(model, x_obs, theta_init, epis = 0.001, S = 100):
    # vectorized, draw multiple MC chains
    # epis: step size
    # S: length of each chain
    # theta_init: dim m*d
    model.to(device)
    
    theta0 = theta_init.view(-1, 3).to(device) # initial value of theta
    x_obs = x_obs.to(device).view(1, -1)
    epis = epis.to(device).view(1, -1)

    n_chain = theta_init.shape[0]
    samples = torch.zeros(n_chain, S, theta_dim)
    
    for i in range(S):
        like_score_hat = model(theta0, x_obs.repeat(theta0.shape[0], 1)).detach().to(device)
        prior_score = 0.0
        theta1 = theta0 + epis * (like_score_hat + prior_score) + torch.sqrt(2.0 * epis) * torch.randn(theta0.shape).to(device) # draw a new sample
        theta1 = proj_vec(theta1, x_obs)
        theta0 = theta1 

        samples[:, i, :] = theta1.cpu().clone()
    return samples



def main():
    # ensure the foler "res_nmodel" exists
    Path("res_nmodel").mkdir(exist_ok=True)

    for obs_id in range(100):
        task_id = obs_id // 10
        hidden_size = 64
        num_layers = 2


        model = Tanh_nmodel_LikeScoreMatchingNN(
            theta_dim, x_dim, obs_size, hidden_size, num_layers
        ).to(device)


        ckpt_path = Path("nmodel_init") / f"checkpoint_task{task_id}.pth"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()


        x_obs = pd.read_csv(f"data_obs/x_obs_task{obs_id}.csv")
        x_obs = torch.tensor(x_obs.values, dtype = torch.float32).contiguous()

        epis = 1 / obs_size * torch.ones(3) 
        epis[0] *= 0.01
        epis[2] *= 0.001

        S = 5000
        n_chain = 10
        theta_init = torch.tensor([a1, a2, a3]) + torch.rand(n_chain, 3) * ( torch.tensor([b1, b2, b3]) - torch.tensor([a1, a2, a3]) )

        samples = proj_draw_post_vec(model, x_obs, theta_init, epis, S)

        np.save(f"res_nmodel/samples_obs{obs_id}.npy", samples.cpu().numpy())


if __name__ == "__main__":
    main()