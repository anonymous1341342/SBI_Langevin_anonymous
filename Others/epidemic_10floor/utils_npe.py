import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
import yaml
import numpy as np
import pandas as pd
import glob
from sklearn.datasets import make_moons
import torch
from torch.nn import Linear, ReLU
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# No prior boundary, we use log_theta

# Settings for the floor and room assignments
K = 10 # number of floors
N = 600
NR = 2 # number of people in each room
NF = int(N/K) # number of people on each floor

F_assign = torch.zeros(N, K)
for k in range(K):
    F_assign[(k*NF):((k+1)*NF), k] = 1
C_F = F_assign @ F_assign.T 

R_assign = torch.zeros(N, int(N/NR))
for r in range( int(N/NR) ):
    R_assign[(r*NR):((r+1)*NR), r] = 1
C_R = R_assign @ R_assign.T

F_assign = F_assign.to(device)
C_F = C_F.to(device)
C_R = C_R.to(device)

gamma = 0.05
alpha = 0.1
eta = 0.1 # 0.5
T = 52


# class DataModule(L.LightningDataModule):
#     def __init__(self, dataset, seed, batch_size, train_frac):
#         super().__init__()
#         self.dataset = dataset
#         self.seed = seed
#         self.batch_size = batch_size
#         self.train_frac = train_frac
# 
#     
#     def setup(self, stage):
#         train_size = int(self.train_frac * len(self.dataset))
#         val_size = len(self.dataset) - train_size
#         self.train, self.val = random_split(
#                 self.dataset,
#                 (train_size, val_size),
#                 torch.Generator().manual_seed(self.seed)
#         )
#         self.val_size = val_size
# 
#     def train_dataloader(self):
#         return DataLoader(self.train, self.batch_size, shuffle=True)
#     
#     def val_dataloader(self):
#         return DataLoader(self.val, self.val_size)
    
def lower_tri(values, dim):
    if values.shape[0] > 1:
        L = torch.zeros(values.shape[0], dim, dim, device=values.device)
        tril_ix = torch.tril_indices(dim, dim)
        L[:, tril_ix[0], tril_ix[1]] = values
    # special case for non-batched inputs
    else:
        L = torch.zeros(dim, dim, device=values.device)
        tril_ix = torch.tril_indices(dim, dim)
        L[tril_ix[0], tril_ix[1]] = values[0]
    return L

def diag(values):
    if values.shape[0] > 1:
        L = torch.diag_embed(values)
    # special case for non-batched inputs
    else:
        L = torch.diag(values[0])
    return L

def contact_matrix(arr):
    x, y = np.meshgrid(arr, arr)
    return (x == y).astype(int)

def save_results(posterior_params, val_losses, cfg):
    if cfg.simulator.name in ["si-model", "crkp"]:
        mu = posterior_params[0].item()
        sigma = posterior_params[1].item()
        print(np.round(mu, 3))
        print(np.round(sigma, 3))
        prior_mu = cfg.simulator["prior_mu"]
        prior_sigma = cfg.simulator["prior_sigma"]
    else:
        mu = posterior_params[0].tolist()
        L = posterior_params[1]
        sigma = (L @ L.T).tolist()
        sdiag = (L @ L.T).diag().tolist()
        print(np.round(mu, 3))
        print(np.round(sdiag, 3)) # marginal variances
        prior_mu = cfg.simulator["prior_mu"]
        prior_sigma = cfg.simulator["prior_sigma"]
    results = {"mu": mu, "sigma":sigma,
               "val_loss": val_losses[-1],
               "n_sample": cfg.simulator["n_sample"],
               "batch_size": cfg.train["batch_size"],
               "N": cfg.simulator["N"],
               "prior_mu": prior_mu,
               "prior_sigma": prior_sigma,
               "name": cfg.simulator["name"]}
    for key in cfg["model"]:
        results[key] = cfg["model"][key]
    # should probably save seed, etc.
    with open("results.yaml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(results, yaml_file)
        
# reading multiruns

def get_results(path, drop=True, multirun=True):
    extension =  "/results.yaml"
    if multirun: extension = "/**" + extension
    # if multirun:
    #     extension = "/**/results.yaml"
    # else:
    #     extension = "/results.yaml"
    results = glob.glob(path + extension)
    data = dict()
    for res in results:
        with open(res, "r") as stream:
            yml = yaml.safe_load(stream)
            for k, v in yml.items():
                if k not in data.keys():
                    data[k] = [v]
                else:
                    data[k].append(v)
    data = pd.DataFrame(data)
    data.drop(columns=["_target_", "lr", "batch_size", "dropout", "seed"])
    return data
        
# LIKELIHOOD BASED ESTIMATION

def simulator(alpha, beta, gamma, N, T, seed, het=False):
    if not het:
        beta = [beta]
    X  = np.empty((N, T))
    np.random.seed(seed)
    X[:, 0] = np.random.binomial(1, alpha, N)
    F = np.arange(N) % 5
    R = np.arange(N) % (N // 2)
    fC = contact_matrix(F)
    rC = contact_matrix(R)
    for t in range(1, T):
        I = X[:, t-1]
        # components dependent on individual covariates
        hazard = compute_hazard(beta, I, N, F, fC, rC, het)
        p = 1 - np.exp(-hazard)
        new_infections = np.random.binomial(1, p, N)
        X[:, t] = np.where(I, np.ones(N), new_infections)
        discharge = np.random.binomial(1, gamma, N)
        screening = np.random.binomial(1, alpha, N)
        X[:, t] = np.where(discharge, screening, X[:, t])
    return X

def compute_hazard(beta, I, N, F, fC, rC, het):
    hazard = I.sum() * beta[0] * np.ones(N) / N
    if het:
        hazard += (fC * I).sum(1) * beta[F+1] / 60
        hazard += (rC * I).sum(1) * beta[-1] / 2
    return hazard

def nll(beta, alpha, gamma, N, T, X, het):
    # beta = beta / np.array([1, 300, 300, 300, 300, 300, 300])
    return - x_loglikelihood(beta, alpha, gamma, N, T, X, het)

def x_loglikelihood(beta, alpha, gamma, N, T, X, het=False):
    ans = np.log(
        alpha ** X[:, 0] * (1 - alpha) ** (1 - X[:, 0])
        ).sum()
    if not het:
        beta = [beta]
    F = np.arange(N) % 5
    R = np.arange(N) % (N // 2)
    fC = contact_matrix(F)
    rC = contact_matrix(R)
    for t in range(1, T):
        xs = X[:, t-1]
        xt = X[:, t]
        hazard = compute_hazard(beta, xs, N, F, fC, rC, het)
        ans += (xt * xs  * np.log(
            gamma * alpha + (1 - gamma)
        )).sum()
        ans += (xt * (1 - xs)  * np.log(
            gamma * alpha + (1 - gamma) * (1 - np.exp(- hazard))
        )).sum()
        ans += ((1 - xt) * xs  * np.log(
            gamma * (1 - alpha) + 1e-8
        )).sum()
        ans += ((1 - xt) * (1 - xs) * np.log(
            gamma *(1 - alpha) + (1 - gamma) * (np.exp(- hazard))
        )).sum()
    return ans


### misc

def lognormal_sd(log_mean, log_sd):
    a = np.exp(log_sd**2) - 1
    b = np.exp(2*log_mean + log_sd**2)
    return (a*b)**0.5

class MoonsDataset(Dataset):
    def __init__(self, n_sample, random_state):
        self.n_sample = n_sample
        self.random_state = random_state
        self.data = self._make_data()

    def _make_data(self):
        arr = make_moons(self.n_sample, noise=0.05, random_state=self.random_state)[0]
        return torch.from_numpy(arr).float()

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, index):
        return torch.empty(0), self.data[index]


class GaussianDensityNetwork(L.LightningModule):
    def __init__(self, d_x, d_theta, d_model, lr, weight_decay,
                 mean_field):
        super().__init__()
        self.name = "gdn"
        # compute number of outputs
        self.dim = d_theta
        # assume diagonal covariance matrix
        if mean_field:
            n_outputs = self.dim * 2
        else:
            n_outputs = self.dim + self.dim*(self.dim + 1) // 2
        self.ff = torch.nn.Sequential(
            Linear(d_x, d_model),
            ReLU(),
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, n_outputs),
        )
        # eventually need to save this as an hparam if i am checkpointing models
        self.lr = lr
        self.wd = weight_decay
        self.mean_field = mean_field
        self.val_losses = []

    def forward(self, x):
        assert len(x.shape) == 2
        y = self.ff(x)
        mu = y[:, :self.dim]
        sigma = y[:, self.dim:]
        # case one: unidimensional or mean field
        if self.dim == 1:
            sigma = torch.exp(sigma)
        elif self.mean_field:
            sigma = diag(torch.exp(sigma))
        else:
            sigma = lower_tri(sigma, self.dim)
            # force diagonal entries to be positive
            sigma.diagonal(dim1=-2, dim2=-1).copy_(
                sigma.diagonal(dim1=-2,dim2=-1).exp()
            )
        return mu, sigma
    
    def training_step(self, batch, batch_idx):
        x, theta = batch
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, theta = batch
        assert len(theta.shape) > 1
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

    def gaussiannll(self, theta, mu, sigma):
        p = self.dim
        if p == 1:
            normal = Normal(mu, sigma)
            l = - normal.log_prob(theta)
        else:
            L = sigma
            mvn = MultivariateNormal(loc=mu, scale_tril=L)
            l = - mvn.log_prob(theta)

        return l.mean()
    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
    

    def predict_step(self, x):
        # this returns standard deviation
        mu, sigma = self(x)
        return mu, sigma
    
    # TODO would it make sense to have a "sample" method?


#######################################################
#               Generate reference table              #
#######################################################
def get_SS(y):
    """
        get the 364-dimensional summary statistics
    """
    all_rate = y.mean(dim = 0)
    floor_rates = y.reshape(K, NF, T).mean(dim = 1)
    room_rate = ( (y.reshape(-1, NR, T).sum(dim = 1) > 1.5) * 1.0 ).mean(dim = 0) # rate that both roommates are infected
    return torch.cat( (all_rate, floor_rates.ravel(), room_rate), dim = 0 )

def gen_ref_log(mean_theta, std_theta, sample_size = 10000):
    """
        generate theta from a (truncated) gaussian proposal distribution, and then use theta to generate x
        mean_theta: the mean of the truncated normal, a 1-dim tensor of the same length as theta
        std_theta: the std of the truncated normal, a 1-dim tensor of the same length as theta
        lower: the lower bound for each dimension of theta, a 1-dim tensor
        upper: the upper bound for each dimension of theta, a 1-dim tensor
    """
    # use inverse sampling to draw theta from truncated normal
    mu_new = mean_theta.view(1, -1).repeat(sample_size, 1)
    sigma_new = std_theta.view(1, -1).repeat(sample_size, 1)

    # draw theta_r0
    log_theta = mu_new + sigma_new * torch.randn(mu_new.shape).to(device)
    log_theta = log_theta.to(device)
    theta = log_theta.exp()
    
    # draw data_r0
    data = torch.zeros(sample_size, (K+2) * T).to(device)
    for i in range(sample_size):
        z = gen_z(N, T)
        y = m_vec_partial(N, T, theta[i], gamma, alpha, eta, F_assign, C_F, C_R, z, NF, NR)
        data[i] = get_SS(y)
    return log_theta, data

#####################
#  Data Generation  #
#####################
def Ind(t):
    """
        Indicator function
    """
    return ( 1.0 * (t > 0) )

def m_vec_partial(N, T, beta, gamma, alpha, eta, F_assign, C_F, C_R, Z, NF, NR):
    """
        generate data in the fully observed case (Algorithm 3 in the paper)
    """
    # eta is the probability that the infection has symptons (can be observed)
    # Z = [allD, allU, allV]: latent variables, allD of dimension N by T is indicator of replacement
    # allU of dimension N by T is for bernoulli sampling, allV of dimension N by T is for bernoulli sampling
    allD, allU, allV = Z
    allD, allU, allV = allD.to(device), allU.to(device), allV.to(device)
    X = torch.zeros(N, T).to(device)
    Y = torch.zeros(N, T).to(device)
    X[:, 0] = Ind(alpha - allU[:, 0]) # initialization
    Y[:, 0] = X[:, 0] # patients are screened when they enter
    for t in range(1, T):
        # First, update X
        D = allD[:, t] # discharge or not
        X[D == 1, t] = Ind(alpha - allU[D == 1, t]) # for replaced patients
        norep_infec_id = torch.logical_and(D == 0, X[:, t-1] > 0.5) # people who are not replaced and have already been infected
        norep_sus_id = torch.logical_and(D == 0, X[:, t-1] < 0.5) # people who are not replaced and are not infected

        X[norep_infec_id, t] = 1 # X[norep_infec_id, t-1]
        lam = hazard(beta, X[:, t-1], F_assign, C_F, C_R, N, NF, NR)
        lam = lam[norep_sus_id]
        X[norep_sus_id, t] = Ind( (1 - (-lam).exp()) - allU[norep_sus_id, t] )

        # Next, get Y based on X
        Y[D == 1, t] = X[D == 1, t]
        id1 = torch.logical_and(X[:, t] > 0.5, Y[:, t-1] < 0.5)
        Y[torch.logical_and(D == 0, id1), t] = Ind(eta - allV[torch.logical_and(D == 0, id1), t])
        Y[torch.logical_and(D == 0, ~id1), t] = Y[torch.logical_and(D == 0, ~id1), t - 1]
    return Y # , X

def hazard(beta, X, F_assign, C_F, C_R, N, NF, NR):
    """
    Calculate the hazard function based on the previous state X_{t-1} and the contact matrices C_F and C_R
    """
    # Input:
    # beta0, beta_middle, beta_last: the parameters, beta_middle is a K-dimensional vector, the other two are scalars
    # X: X_{t-1}, a N dimensional vector
    # F_assign: assignment of floor, a N by K matrix, F_assign[i, k] = individual i lives on floor k
    # C_F: contact matrix of floor, C_F[i, j] = 1{i and j are on the same floor}
    # C_R: contact matrix of room, C_R[i, j] = 1{i and j are in the same room}
    # C_F can be calculated from F_assign, but we make them input to save calculation, cause they are unchanged throughout the dynamics
    # N, NF, NR: scale factors for beta
    # Output:
    # lambda: lambda(t) = (lambda_1(t), ..., lambda_N(t)), recording the hazard for each individual
    N = X.shape[0]
    beta0 = beta[0] / N
    beta_middle = beta[1:-1] / NF
    beta_last = beta[-1] / NR

    return ( beta0 * torch.ones(N, N).to(device) + (F_assign @ beta_middle).view(-1, 1).repeat(1, N) * C_F + beta_last * C_R ) @ X
    # return ( beta0 * torch.ones(N, N) + beta_last * C_R ) @ X

def gen_z(N, T):
    """
        generate the latent random variable
    """
    return [torch.bernoulli(gamma * torch.ones(N, T)).to(device), torch.rand(N, T).to(device), torch.rand(N, T).to(device)]
