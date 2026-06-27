import gc
from utils_SDEMEM import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import shutil
from pathlib import Path

from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.inference import NPE


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
num_epochs = 200 # config["num_epochs"]
batch_size = 200 # int(config["batch_size"]) 
summary_dim = 128 # int(config["summary_dim"]) 
embed_hidden_size = 128 # int(config["embed_hidden_size"]) 
embed_num_layers = 3 # int(config["embed_num_layers"]) 

sample_size = int(1.05e6) # int(config["sample_size"]) 

learning_rate = 3e-4


def main(task_id):
    #################
    # Training Data #
    #################
    # Load SW data
    theta_SW1 = np.load(f"res_SW1/theta_SW1_task{task_id}.npy")[:100]
    theta_SW1 = torch.tensor(theta_SW1, dtype=torch.float32)

    prop_mean = theta_SW1.mean(dim=0, keepdims=True)
    prop_std = theta_SW1.std(dim=0, keepdims=True)

    # Inflate the proposal std
    prop_std *= 2

    prop_std = prop_std.clamp_min(1e-8)
    print(f"Using prop_std = {prop_std}", flush=True)

    # ===== ref_R ===== #
    # "extra" means each simulation is a dataset of size n = obs_size
    orig_path_theta = Path(f"ref_NPE/theta_r0_extra_task{task_id}.npy")
    orig_path_x = Path(f"ref_NPE/x_r0_extra_task{task_id}.npy")


    ########################################
    # Copy training data to node-local /tmp
    ########################################
    # Use TMPDIR if available; otherwise fall back to /tmp.
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    local_dir = tmp_root / f"npe_task_{task_id}"
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path_theta = local_dir / orig_path_theta.name
    local_path_x = local_dir / orig_path_x.name

    copy_start_time = time.time()

    if not local_path_theta.exists():
        print(f"Copying {orig_path_theta} -> {local_path_theta}", flush=True)
        shutil.copy2(orig_path_theta, local_path_theta)

    if not local_path_x.exists():
        print(f"Copying {orig_path_x} -> {local_path_x}", flush=True)
        shutil.copy2(orig_path_x, local_path_x)

    copy_end_time = time.time()
    print(
        f"Finished copying training data to local tmp in {copy_end_time - copy_start_time:.2f}s",
        flush=True
    )

    # From this point on, use the local tmp copies for training.
    path_theta = local_path_theta
    path_x = local_path_x


    ########################################
    # Custom Dataset (memory-efficient)
    ########################################
    class NPENpyDataset:
        """
        Memory-efficient dataset backed by mmap .npy files.
        Batches are read directly by slicing the memmap arrays.
        """
        def __init__(self, theta_path, x_path, obs_size, x_dim):
            self.theta_np = np.load(theta_path, mmap_mode="r")
            self.x_np = np.load(x_path, mmap_mode="r")

            assert self.theta_np.shape[0] == self.x_np.shape[0]

            self.obs_size = obs_size
            self.x_dim = x_dim

        def __len__(self):
            return self.theta_np.shape[0]

        def get_batch(self, start, end):
            """
            Read a whole batch from memmap, copy it into regular writable RAM,
            and pin the CPU memory for faster host-to-device transfer.
            """
            theta_np = np.array(self.theta_np[start:end], copy=True)
            x_np = np.array(self.x_np[start:end], copy=True)

            theta_batch = torch.from_numpy(theta_np).pin_memory()
            x_batch = torch.from_numpy(x_np).view(-1, self.obs_size, self.x_dim).pin_memory()

            return theta_batch, x_batch


    ########################################
    # Build density estimator
    ########################################
    def build_npe_model(x_dim, embed_hidden_size, embed_num_layers, summary_dim):
        """
        Build embedding network + MAF posterior network.
        """
        single_trial_net = FCEmbedding(
            input_dim=x_dim,
            num_hiddens=embed_hidden_size,
            num_layers=embed_num_layers,
            output_dim=summary_dim,
        )

        embedding_net = PermutationInvariantEmbedding(
            single_trial_net,
            trial_net_output_dim=summary_dim,
            num_hiddens = embed_hidden_size,
            num_layers = embed_num_layers,
            output_dim = summary_dim,
        )

        # This returns a builder function
        return posterior_nn("maf", embedding_net=embedding_net)


    ########################################
    # Custom training loop
    ########################################
    def train_npe_from_dataset(
        dataset,
        model_builder,
        batch_size,
        num_epochs,
        device,
        lr=5e-4,
        validation_fraction=0.1,
        patience=20,
    ):
        """
        Custom training loop for NPE.
        Read batches directly from memmap instead of using DataLoader.
        Also measure time spent in data loading, device transfer, and computation.
        """

        # ---- deterministic train/val split ----
        n = len(dataset)
        n_val = max(1, int(n * validation_fraction))
        n_train = n - n_val

        # ---- build model using a small batch (needed for sbi standardization) ----
        n_init = min(1024, n_train)
        theta_init, x_init = dataset.get_batch(0, n_init)
        density_estimator = model_builder(theta_init, x_init).to(device)

        optimizer = torch.optim.Adam(density_estimator.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            print(f"Starting epoch {epoch}", flush=True)

            ################################
            # Train
            ################################
            density_estimator.train()
            train_loss_sum = 0.0
            train_n = 0

            # Time breakdown for training
            train_load_time = 0.0
            train_transfer_time = 0.0
            train_compute_time = 0.0

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)

                # Measure data loading time
                t0 = time.time()
                theta_batch, x_batch = dataset.get_batch(start, end)
                t1 = time.time()
                train_load_time += t1 - t0

                # Measure CPU-to-GPU transfer time
                theta_batch = theta_batch.to(device, non_blocking=True)
                x_batch = x_batch.to(device, non_blocking=True)
                t2 = time.time()
                train_transfer_time += t2 - t1

                # Measure forward/backward/update time
                optimizer.zero_grad(set_to_none=True)

                # NPE: learn p(theta | x)
                losses = density_estimator.loss(theta_batch, x_batch)
                loss = losses.mean()

                loss.backward()
                optimizer.step()
                t3 = time.time()
                train_compute_time += t3 - t2

                bs = theta_batch.shape[0]
                train_loss_sum += loss.item() * bs
                train_n += bs

            train_loss = train_loss_sum / train_n

            ################################
            # Validation
            ################################
            density_estimator.eval()
            val_loss_sum = 0.0
            val_n = 0

            # Time breakdown for validation
            val_load_time = 0.0
            val_transfer_time = 0.0
            val_compute_time = 0.0

            with torch.no_grad():
                for start in range(n_train, n, batch_size):
                    end = min(start + batch_size, n)

                    # Measure data loading time
                    t0 = time.time()
                    theta_batch, x_batch = dataset.get_batch(start, end)
                    t1 = time.time()
                    val_load_time += t1 - t0

                    # Measure CPU-to-GPU transfer time
                    theta_batch = theta_batch.to(device, non_blocking=True)
                    x_batch = x_batch.to(device, non_blocking=True)
                    t2 = time.time()
                    val_transfer_time += t2 - t1

                    # Measure validation forward time
                    losses = density_estimator.loss(theta_batch, x_batch)
                    t3 = time.time()
                    val_compute_time += t3 - t2

                    bs = theta_batch.shape[0]
                    val_loss_sum += losses.mean().item() * bs
                    val_n += bs

            val_loss = val_loss_sum / val_n

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            print(
                f"[Epoch {epoch}] "
                f"train: {train_loss:.3f} | val: {val_loss:.3f} | "
                f"time: {epoch_time:.2f}s | "
                f"train_load: {train_load_time:.2f}s | "
                f"train_transfer: {train_transfer_time:.2f}s | "
                f"train_compute: {train_compute_time:.2f}s | "
                f"val_load: {val_load_time:.2f}s | "
                f"val_transfer: {val_transfer_time:.2f}s | "
                f"val_compute: {val_compute_time:.2f}s",
                flush=True,
            )

            ################################
            # Early stopping
            ################################
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0

                # Save the best model state on CPU
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
    # ===== Replace original training ===== #
    ########################################

    # Create dataset WITHOUT loading everything into RAM
    dataset = NPENpyDataset(
        theta_path=path_theta,
        x_path=path_x,
        obs_size=obs_size,
        x_dim=x_dim,
    )

    # Build model builder
    model_builder = build_npe_model(
        x_dim=x_dim,
        embed_hidden_size=embed_hidden_size,
        embed_num_layers=embed_num_layers,
        summary_dim=summary_dim,
    )

    # Train
    density_estimator = train_npe_from_dataset(
        dataset=dataset,
        model_builder=model_builder,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        lr=learning_rate,
        validation_fraction=0.1,
        patience=20,
    )

    # Save model
    save_dir_nn = Path("NPE_embed_model_newdeepsets")
    save_dir_nn.mkdir(parents=True, exist_ok=True)

    model_path = save_dir_nn / f"npe_net_weights_task{task_id}.pth"
    torch.save(density_estimator.state_dict(), model_path)
    print(f"Saved model to {model_path}", flush=True)

    # Release mmap references before deleting files
    del dataset
    del model_builder
    gc.collect()

    ########################################
    # Delete files after successful save
    ########################################
    if model_path.exists():
        # Delete local tmp copies
        if path_theta.exists():
            path_theta.unlink()
            print(f"Deleted local tmp file {path_theta}", flush=True)

        if path_x.exists():
            path_x.unlink()
            print(f"Deleted local tmp file {path_x}", flush=True)

        # Delete original shared-storage files
        if orig_path_theta.exists():
            orig_path_theta.unlink()
            print(f"Deleted original file {orig_path_theta}", flush=True)

        if orig_path_x.exists():
            orig_path_x.unlink()
            print(f"Deleted original file {orig_path_x}", flush=True)
    else:
        print("Model file was not found, so the training data files were kept.", flush=True)

    end_time = time.time()
    print(f"Total time = {(end_time - start_time) / 60:.2f} minutes", flush=True)



if __name__ == "__main__":
    task_id = sys.argv[1]
    main(task_id)