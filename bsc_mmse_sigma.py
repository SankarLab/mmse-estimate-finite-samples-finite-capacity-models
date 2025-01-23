from mg import BSCGaussian
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import LogisticRegression as gdLR
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--p_n', type = float, help='crossover probability')
parser.add_argument('--seed', type = int, help='random seed')
parser.add_argument('-p', type = float, help='prior on S=1')
parser.add_argument('-n', type = int, help='number of training samples')
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

n = args.n
p = args.p
p_n = args.p_n
for sigma in tqdm(np.linspace(0,3,20),leave=False):
    mg.sigma = sigma
    mg.p_n = p_n
    mg.p = p
    #X_train,y_train = mg.generate(1e6,as_df=False)
    X_expected, y_expected = mg.generate(1e6,as_df=False)
    #big_LR = gdLR(loss='square').fit(X_train.reshape(-1,1), y_train,lr=1e-1,opt='AdamW',epochs=10000,scheduler='cosine', verbose=False)
    #delta_a = MSE(big_LR.predict_proba(X_expected.reshape(-1,1)) , mg._prob(X_expected))
    delta_a = MSE(opt(X_expected,1e7), mg._prob(X_expected))
    for seed in tqdm(range(30),leave=False):
        mg.rng = np.random.default_rng(42+seed)
        X,y = mg.generate(n,as_df=False)
        LR = gdLR(loss='square').fit(X.reshape(-1,1), y,lr=1e-1,epochs=5000, opt='AdamW', scheduler='cosine')
        lin_mse = MSE(LR.predict_proba(X.reshape(-1,1)), y)
        sigma_results.loc[len(sigma_results)] = ( lin_mse,delta_a, sigma, seed, mg.mmse_estimate(1e6),delta,n,p,p_n)
sigma_results.to_csv('bsc_sigma_'+str(p_n) + '_' + str(n) + '_'+str(p)+'.csv',mode='a')