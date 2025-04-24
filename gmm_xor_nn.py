import os
from mg import ScalingXORGaussian2D
import numpy as np
import pandas as pd
from models import NeuralNetwork as gdNN
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser
import torch

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

parser = ArgumentParser()
parser.add_argument('-s','--sigma', type = float, help='sanitizing noise', default=-1)
parser.add_argument('--delta', type = float, help='separation of class means', default = 2)
parser.add_argument('--n_modes', type = int, help='number of modes per class', default = 2)
parser.add_argument('--seed', type = int, help='random seed')
parser.add_argument('-n', type = int, help='number of training samples', default = -1)
parser.add_argument('--hidden', type = int, help='number of hidden dimensions', default = 10)
parser.add_argument('--val_n', type = int, help='number of validation data', default = 500)
parser.add_argument('--results_path', type = dir_path, help='directory to store results (must exist)', default = 'results')
parser.add_argument('--model_path', type = dir_path, help='directory to store results (must exist)', default = 'models')

args = parser.parse_args()

def MSE(y_pred: np.ndarray, y_true:np.ndarray) -> float:
    return ((y_pred - y_true)**2).mean()


sigma_results = pd.DataFrame(columns=['mse','delta_A','sigma','seed','mmse','delta','eps_c','n_modes','n','h', 'val_n','val_mse', 'eps_c_tilde'])

n = args.n
delta = args.delta
n_modes = args.n_modes
h = args.hidden
val_n = args.val_n
sigma = args.sigma
n_range = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000] if args.n==-1 else [args.n]
sigma_range = np.linspace(0,5,20) if args.sigma == -1 else [args.sigma]
first = True
for sigma in sigma_range:
    mg = ScalingXORGaussian2D(n_modes_per_class = n_modes, scale=delta, sigma_noise=sigma,seed=args.seed)
    X_expected, y_expected, _ = mg.generate(1e6,as_df=False)
    X_big, y_big, _ = mg.generate(2e6,as_df=False)
    nn = gdNN(loss='square',device='cuda').fit(X_big, y_big,lr=1e-2, scheduler='cosine', epochs=10000, opt='AdamW', n_hidden = h)
    delta_a = MSE(nn.predict_proba(X_expected), mg._prob(X_expected))
    for n in n_range:
        for seed in tqdm(range(30),leave=False):
            mg.rng = np.random.default_rng(args.seed+seed)
            X,y,_ = mg.generate(n,as_df=False)
            X_val, y_val, _ = mg.generate(val_n,as_df=False)
            LR = gdNN(loss='square',device='cuda').fit(X, y,lr=1e-2, scheduler='cosine',epochs=10000,opt='AdamW',n_hidden = h)
            if first:
                file_path = os.path.join(args.model_path,f'gmm_xor_{args.n_modes}_{args.hidden}_{val_n}.pth.tar.gz')
                torch.save(LR.model.state_dict(), file_path,_use_new_zipfile_serialization=True)
                file_size = os.path.getsize(file_path) * 8
                first=False

            lin_mse = MSE(LR.predict_proba(X), y)
            val_mse = MSE(LR.predict_proba(X_val), y_val)
            v_n = ((nn.predict_proba(X) - y)**2).var(ddof=1)
            v_m = ((LR.predict_proba(X_val) - y_val)**2).var(ddof=1)
            eps_c = np.sqrt(2 * v_n *np.log(2/0.05)/n) + 7 * np.log(2/0.05)/(3 * (n-1))
            eps_c_val = np.sqrt(2 * v_m *np.log(6/0.05)/val_n) + 7 * np.log(6/0.05)/(3 * (val_n-1)) + np.sqrt(2 * v_n *np.log(6/0.05)/n) + 7 * np.log(6/0.05)/(3 * (n-1))
            eps_c_tilde = np.sqrt(file_size * np.log(2) + 2 * np.log(file_size) + np.log(3/0.05)) * (1/np.sqrt(2*n) ) + eps_c_val
            sigma_results.loc[len(sigma_results)] = ( lin_mse,delta_a, sigma, seed, mg.mmse_estimate(1e6),delta,eps_c,n_modes,n,h,val_n, val_mse, eps_c_tilde)
sigma_results.to_csv(os.path.join(args.results_path, f'gmm_xor_{args.sigma}_{args.n}_{args.delta}_{args.n_modes}_{args.hidden}_{val_n}.csv'),mode='a')
