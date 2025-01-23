from mg import CCGaussian
import numpy as np
import pandas as pd
from models import LogisticRegression as gdLR
from tqdm.autonotebook import tqdm
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-s','--sigma', type = float, help='sanitizing noise', default=-1)
parser.add_argument('--delta', type = float, help='separation of means', default = 2)
parser.add_argument('--dim', type = int, help='dimensionality of data', default = 2)
parser.add_argument('--seed', type = int, help='random seed')
parser.add_argument('-p', type = float, help='prior on S=1')
parser.add_argument('-n', type = int, help='number of training samples', default = -1)
args = parser.parse_args()

def MSE(y_pred: np.ndarray, y_true:np.ndarray) -> float:
    return ((y_pred - y_true)**2).mean()

def opt(z,n=1e5):
    x,_ = mg.generate(n,as_df=False)
    a = np.cov(x,mg._theta(x).reshape(-1,1),rowvar=False,ddof=1)[-1][:-1]
    a = a @ np.linalg.inv(np.cov(x,rowvar=False,ddof=1))
    b = mg._theta(x).mean() - a.T@ x.mean(axis=0)
    return 1/(1+np.exp(-z@a - b))


sigma_results = pd.DataFrame(columns=['mse','delta_A','sigma','seed','mmse','delta','d','n','p'])

n = args.n
p = args.p
delta = args.delta
d = args.dim
sigma_0 = 1 * np.eye(d)
sigma_1 = 3 * np.eye(d)
mu_0 = np.array([delta] + (d-1)*[0])
mu_1 = - mu_0
sigma = args.sigma
mg = CCGaussian(mu_0,mu_1,sigma_0, sigma_1, sigma=sigma,p=p)
n_range = np.arange(100,1000,100) if args.n==-1 else [args.n]
sigma_range = np.linspace(0,5,20) if args.sigma == -1 else [args.sigma]
for sigma in sigma_range:
    for n in n_range:
        mg.rng = np.random.default_rng(args.seed)
        mg.sigma = sigma
        mg.p = p
        mg.sigma_0 = sigma_0
        mg.sigma_1 = sigma_1
        mu_0 = np.array([delta] + (d-1)*[0])
        mu_1 = - mu_0
        mg.mu_1 = mu_1
        mg.mu_0 = mu_0
        X_expected, y_expected = mg.generate(1e6,as_df=False)
        delta_a = MSE(opt(X_expected, 1e7), mg._prob(X_expected))
        for seed in tqdm(range(30),leave=False):
            mg.rng = np.random.default_rng(args.seed+seed)
            X,y = mg.generate(n,as_df=False)
            LR = gdLR(loss='square',device='cuda').fit(X, y,lr=1e-1, scheduler='cosine',epochs=5000,opt='AdamW')
            lin_mse = MSE(LR.predict_proba(X), y)
            sigma_results.loc[len(sigma_results)] = ( lin_mse,delta_a, sigma, seed, mg.mmse_estimate(1e6),delta,d,n,p)
sigma_results.to_csv('gaussian_spherical_'+str(args.sigma)+'_'+str(args.n)+'_'+str(args.delta) + '_' + str(args.p) + '_' + str(args.dim)+ '.csv',mode='a')
