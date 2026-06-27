from utils_monoBP_single import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import pandas as pd
from pathlib import Path
from sbi.inference import NLE




start_time = time.time()

# =============== Load configurations =============== #
# config = json.loads(sys.argv[2])
num_epochs = 1 # config["num_epochs"] 
batch_size = 200 # int(config["batch_size"])

ref_size = 2e8 # int(config["ref_size"]) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma = 0.1 # noise level
obs_size = 1000
# prior for theta0
a0 = -5.0
b0 = 5.0
# prior for theta1-thetaM
a = 0.0
b = 1.0


def main(task_id):
    ##### Read Localization results
    theta_pre = torch.tensor(pd.read_csv(f"res_SW1_precond/theta_pre_task{task_id}.csv").values, dtype = torch.float32).contiguous()


    ##### generate training data
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


    # generate reference data in batches to avoid memory issues
    sample_size = ref_size
    total_size = sample_size + 1000
    n_batches = 10
    batch_size_gen = math.ceil(total_size / n_batches)

    theta_r0 = torch.empty(total_size, M + 1)
    data_r0 = torch.empty(total_size, 2)

    for b in range(n_batches):
        start = b * batch_size_gen
        end = min((b + 1) * batch_size_gen, total_size)
        cur_size = end - start

        theta_b, data_b = gen_ref_distinct_theta(
            mean_theta, std_theta, lower, upper, cur_size
        )

        theta_r0[start:end] = theta_b
        data_r0[start:end] = data_b

        print(f"generated batch {b + 1}/{n_batches}: {start} to {end}")




    bad_mask = torch.isinf(theta_r0).any(dim=1)
    theta_r0 = theta_r0[~bad_mask]
    data_r0 = data_r0[~bad_mask]

    print("Inflation rate used:", inf_rate)
    print("Actual inflation rate:", theta_r0.std(dim = 0) / theta_pre.std(dim = 0))
    print(f"number of data points having inf = {bad_mask.sum()}")        
    print(f'Shape of theta and x: {theta_r0.shape}, {data_r0.shape}')


    # ====== Train the model ====== #
    inference = NLE(show_progress_bars=True, density_estimator="maf", device = device)
    inference.append_simulations(
        theta_r0, 
        data_r0,
        data_device='cpu').train(
            training_batch_size=batch_size,
            validation_fraction=0.5, # use 50% of the data for validation
            show_train_summary=True,
            max_num_epochs=num_epochs,
            stop_after_epochs=10
            )

    print(inference._summary)


    # Save the NLE model
    save_dir_nn = Path(f"NLE_model")
    save_dir_nn.mkdir(parents=True, exist_ok=True)  # create folder if missing
    torch.save(inference._neural_net.state_dict(), save_dir_nn / f"nle_net_weights_task{task_id}.pth")

    end_time = time.time()
    print(f'Total time = {(end_time - start_time) / 60:.2f} minutes')



if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)