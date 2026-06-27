import gc
from utils_SDEMEM import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import json
from pathlib import Path

from sbi.neural_nets import likelihood_nn


start_time = time.time()

# ===== Setting ===== #
obs_size = 200
T = 30
theta_dim = 12
x_dim = 180

prior_mean = torch.tensor([5, 1, 3, -1.5, -0.694, -3, 0.027, 0, -0.8, -0.8, -0.8, -0.8], dtype = torch.float32)
prior_std = torch.tensor([1, 1, 1, 1, 0.6, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5], dtype = torch.float32)

# [log_m0, log_scale, log_offset, log_sigma, mu_delta, mu_gamma, mu_k, mu_t0, log_tau_delta, log_tau_gamma, log_tau_k, log_tau_t0]
theta_true = torch.tensor([5.7, 0.7, 2.08, -1.6, -0.694, -3, 0.027, 0, -1.15, -1.15, -1.15, -1.15], dtype = torch.float32).reshape(1, -1)


# =============== Load configurations =============== #
num_epochs = 2 # config["num_epochs"] 
batch_size = 200 # int(config["batch_size"]) 
ref_size = int(2.1e8) # int(config["ref_size"]) 


def main(task_id):
    # ===== Load SW proposal information ===== #
    theta_SW1 = np.load(f"res_SW1/theta_SW1_task{task_id}.npy")[:100]
    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)

    prop_mean = theta_SW1.mean(dim = 0, keepdims = True)
    prop_std = theta_SW1.std(dim = 0, keepdims = True) 

    # inflate the proposal std
    prop_std *= 2


    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}")

    # ===== Paths for saved training data ===== #
    path_theta = Path(f"ref_NLE/theta_r0_task{task_id}.npy")
    path_x = Path(f"ref_NLE/x_r0_task{task_id}.npy")


    ########################################
    # Custom Dataset (memory-efficient)
    ########################################
    class NLENpyDataset:
        """
        Memory-efficient dataset backed by mmap .npy files.
        Batches are read directly by slicing the memmap arrays.
        """
        def __init__(self, theta_path, x_path):
            self.theta_np = np.load(theta_path, mmap_mode="r")
            self.x_np = np.load(x_path, mmap_mode="r")

            assert self.theta_np.shape[0] == self.x_np.shape[0]

        def __len__(self):
            return self.theta_np.shape[0]

        def get_batch(self, start, end):
            """
            Read a whole batch directly from memmap.
            """
            theta_batch = torch.from_numpy(
                np.asarray(self.theta_np[start:end], dtype=np.float32)
            )

            x_batch = torch.from_numpy(
                np.asarray(self.x_np[start:end], dtype=np.float32)
            )

            return theta_batch, x_batch


    ########################################
    # Build density estimator
    ########################################
    def build_nle_model():
        """
        Build a likelihood estimator using MAF.
        """
        return likelihood_nn(model="maf")


    ########################################
    # Custom training loop
    ########################################
    def train_nle_from_dataset(
        dataset,
        model_builder,
        batch_size,
        num_epochs,
        device,
        lr=5e-4,
        validation_fraction=0.1,
        patience=10,
        init_batch_size=1024,
    ):
        """
        Custom training loop for NLE.
        Read batches directly from memmap instead of using DataLoader.
        """

        # Deterministic train/val split
        n = len(dataset)
        n_val = max(1, int(n * validation_fraction))
        n_train = n - n_val

        # Build model using a small batch
        # This is needed because sbi computes standardization statistics here
        n_init = min(init_batch_size, n_train)
        theta_init, x_init = dataset.get_batch(0, n_init)
        density_estimator = model_builder(theta_init, x_init).to(device)

        optimizer = torch.optim.Adam(density_estimator.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, num_epochs + 1):
            print(f"Starting epoch {epoch}", flush=True)

            ################################
            # Train
            ################################
            density_estimator.train()
            train_loss_sum = 0.0
            train_n = 0

            num_train_batches = (n_train + batch_size - 1) // batch_size

            for batch_idx, start in enumerate(range(0, n_train, batch_size), start=1):
                time1 = time.time()
                end = min(start + batch_size, n_train)

                theta_batch, x_batch = dataset.get_batch(start, end)
                theta_batch = theta_batch.to(device)
                x_batch = x_batch.to(device)

                optimizer.zero_grad(set_to_none=True)

                # NLE learns p(x | theta)
                losses = density_estimator.loss(x_batch, condition=theta_batch)
                loss = losses.mean()

                loss.backward()
                optimizer.step()

                bs = theta_batch.shape[0]
                train_loss_sum += loss.item() * bs
                train_n += bs

                time2 = time.time()
                if batch_idx % 1000 == 0 or batch_idx == num_train_batches:
                    print(
                        f"Epoch {epoch} | train batch {batch_idx}/{num_train_batches} | "
                        f"current loss = {loss.item():.3f} | time = {time2 - time1:.4f}s",
                        flush=True,
                    )

            train_loss = train_loss_sum / train_n

            ################################
            # Validation
            ################################
            density_estimator.eval()
            val_loss_sum = 0.0
            val_n = 0

            num_val_batches = (n_val + batch_size - 1) // batch_size

            with torch.no_grad():
                for batch_idx, start in enumerate(range(n_train, n, batch_size), start=1):
                    end = min(start + batch_size, n)

                    theta_batch, x_batch = dataset.get_batch(start, end)
                    theta_batch = theta_batch.to(device)
                    x_batch = x_batch.to(device)

                    losses = density_estimator.loss(x_batch, condition=theta_batch)
                    batch_loss = losses.mean().item()

                    bs = theta_batch.shape[0]
                    val_loss_sum += batch_loss * bs
                    val_n += bs

                    if batch_idx % 1000 == 0 or batch_idx == num_val_batches:
                        print(
                            f"Epoch {epoch} | val batch {batch_idx}/{num_val_batches} | "
                            f"current loss = {batch_loss:.6f}",
                            flush=True,
                        )

            val_loss = val_loss_sum / val_n

            print(f"[Epoch {epoch}] train: {train_loss:.6f} | val: {val_loss:.6f}", flush=True)

            ################################
            # Early stopping
            ################################
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in density_estimator.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            density_estimator.load_state_dict(best_state)

        return density_estimator


    ########################################
    # Train with memory-friendly pipeline
    ########################################
    dataset = NLENpyDataset(
        theta_path=path_theta,
        x_path=path_x,
    )

    model_builder = build_nle_model()

    density_estimator = train_nle_from_dataset(
        dataset=dataset,
        model_builder=model_builder,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        lr=5e-4,
        validation_fraction=0.1,
        patience=10,
        init_batch_size=1024,
    )

    # Save model
    save_dir_nn = Path("NLE_model")
    save_dir_nn.mkdir(parents=True, exist_ok=True)

    model_path = save_dir_nn / f"nle_net_weights_task{task_id}.pth"
    torch.save(density_estimator.state_dict(), model_path)
    print(f"Saved model to {model_path}", flush=True)

    # Release mmap references before deleting .npy files
    del dataset
    del model_builder
    gc.collect()

    # Delete saved .npy training files only if model save succeeded
    if model_path.exists():
        if path_theta.exists():
            path_theta.unlink()
            print(f"Deleted {path_theta}", flush=True)

        if path_x.exists():
            path_x.unlink()
            print(f"Deleted {path_x}", flush=True)
    else:
        print("Model file was not found, so the .npy files were kept.", flush=True)

    end_time = time.time()
    print(f"Total time = {(end_time - start_time) / 60:.2f} minutes", flush=True)



if __name__ == "__main__":
    task_id = int(sys.argv[1])  
    main(task_id)