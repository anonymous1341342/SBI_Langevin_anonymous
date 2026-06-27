from utils_queuing_single import *
import pandas as pd
import ot
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def ABC_W1(data_obs, theta_set):
    """
    ABC by comparing W1(data_obs, data_simu)
    """
    # data_obs is actually SS_obs
    data_obs = data_obs.to(device)
    theta_set = theta_set.to(device)
    W1_set = torch.zeros(theta_set.shape[0])
    
    for i in range(theta_set.shape[0]):
        # generate simulated data based on theta_i
        theta = theta_set[i]
        data_simu = gen_obs_data(theta[0].item(), theta[1].item(), theta[2].item(), dim = 5, obs_size = 500).to(device)
    
        # calculate W1(data_obs, data_simu)
        marg1 = (1/data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device) # marginal distribution
        marg2 = (1/data_simu.shape[0]) * torch.ones(data_simu.shape[0]).to(device)
        cost_mat = ot.dist(data_obs, data_simu, metric='euclidean')  
        W1_set[i] = ot.emd2(marg1, marg2, cost_mat)
    return W1_set


def main(obs_id):
    # load the observed data
    x_obs = pd.read_csv(f"data_obs/x_obs_task{obs_id}.csv")
    x_obs = torch.tensor(x_obs.values, dtype = torch.float32).contiguous()

    # generate reference table
    sample_size = 20000 # 10000
    theta1 = np.random.uniform(low = a1, high = b1, size = sample_size)
    theta2 = np.random.uniform(low = a2, high = b2, size = sample_size)
    theta3 = np.random.uniform(low = a3, high = b3, size = sample_size)

    theta_r0 = np.c_[theta1, theta2, theta3]
    theta_r0 = torch.tensor(theta_r0, dtype = torch.float32)
    theta_set = theta_r0

    # run ABC
    W1_set = ABC_W1(x_obs, theta_set)
    smallest_values, smallest_indices = torch.topk(W1_set, 200, largest=False)
    theta_r1 = theta_set[smallest_indices].clone()

    # create the folder "res" if it does not exist
    Path("res").mkdir(exist_ok=True)

    # save the result
    df = pd.DataFrame( theta_r1.cpu().numpy() )
    df.to_csv(f"res/ABCW1_x_obs_{obs_id}.csv", index=False)

if __name__ == "__main__":
    obs_id = sys.argv[1]
    main(obs_id)