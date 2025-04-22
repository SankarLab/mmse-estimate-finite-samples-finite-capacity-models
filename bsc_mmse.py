
from mg import BSCGaussian
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import LogisticRegression as gdLR
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-s', '--sigma', type = float, help='sanitizing noise', default = -1)
parser.add_argument('--seed', type = int, help='random seed')
parser.add_argument('-p', type = float, help='prior on S=1')
parser.add_argument('--p_n', type = float, help='crossover probability')
parser.add_argument('-n', type = int, help='number of training samples', default = -1)
args = parser.parse_args()

def MSE(y_pred: np.ndarray, y_true:np.ndarray) -> float:
    return ((y_pred - y_true)**2).mean()

def opt(z,n=1e5):
    x,_ = mg.generate(n,as_df=False)
    a = np.cov(x.reshape(-1,1),mg._theta(x).reshape(-1,1),rowvar=False,ddof=1)[0][1]/x.var()
    b = mg._theta(x).mean() - a* x.mean()
    return 1/(1+np.exp(-a*z - b))

delta = 1
mg = BSCGaussian(delta,0.3,1/2,1/4)
mg.rng = np.random.default_rng(args.seed)

sigma_results = pd.DataFrame(columns=['mse','delta_A','sigma','seed','mmse','mu','n','p','p_n'])

p = args.p
n_range = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000] if args.n==-1 else [args.n]
sigma_range = np.linspace(0,5,20) if args.sigma == -1 else [args.sigma]
p_n_range = np.linspace(0,1/2,20) if args.p_n == -1 else [args.p_n]
for n in n_range:
    for sigma in sigma_range:
        for p_n in p_n_range:
            mg.sigma = sigma
            mg.p_n = p_n
            mg.p = p
            X_expected, y_expected = mg.generate(1e6,as_df=False)
            delta_a = MSE(opt(X_expected,1e7), mg._prob(X_expected))
            for seed in tqdm(range(30),leave=False):
                mg.rng = np.random.default_rng(42+seed)
                X,y = mg.generate(n,as_df=False)
                LR = gdLR(loss='square').fit(X.reshape(-1,1), y,lr=1e-1,epochs=5000, opt='AdamW', scheduler='cosine')
                lin_mse = MSE(LR.predict_proba(X.reshape(-1,1)), y)
                sigma_results.loc[len(sigma_results)] = ( lin_mse,delta_a, sigma, seed, mg.mmse_estimate(1e6),delta,n,p,p_n)
sigma_results.to_csv('bsc_pn_'+str(args.sigma) + '_' + str(args.n) + '_'+str(args.p)+'.csv',mode='a')
