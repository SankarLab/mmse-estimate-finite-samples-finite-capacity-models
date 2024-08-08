import numpy as np
import pandas as pd
import scipy.stats as st

class MixtureGaussian:
    def __init__(self, delta_D: np.ndarray, delta_C: np.ndarray, sigma: np.ndarray, mu: np.ndarray | float =0, pi_0:float = 1/4, seed:int=42):
        # Given
        self.delta_D = np.array(delta_D)
        self.delta_C = np.array(delta_C)
        self.sigma = np.array(sigma)
        self.mu = np.array(mu)
        self.pi_0 = float(pi_0)
        self.rng = np.random.default_rng(int(seed))

        # Calculated from definition of mu and deltas
        self.mu_1R = self.mu + self.delta_D / 2 - self.delta_C /2
        self.mu_1G = self.mu + self.delta_D / 2 + self.delta_C /2
        self.mu_0R = self.mu - self.delta_D / 2 - self.delta_C /2
        self.mu_0G = self.mu - self.delta_D / 2 + self.delta_C /2
        self.mu_1 = 2* self.pi_0 * (self.mu_1G) +(1-2*pi_0)* (self.mu_1R)
        self.mu_0 = 2* self.pi_0 * (self.mu_0R) +(1-2*pi_0)* (self.mu_0G)
        self.delta_bar = self.mu_1 - self.mu_0

        self._X = None
        self._y = None
        self._group = None
    
    def generate(self, n: int, as_df: bool = True) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_maj = int(n * (1/2 - self.pi_0))
        n_min = int((n - 2*n_maj) // 2)

        X_1R  = self.rng.multivariate_normal(self.mu_1R, self.sigma, size =n_maj)
        X_0G  = self.rng.multivariate_normal(self.mu_0G, self.sigma, size =n_maj)
        X_1G  = self.rng.multivariate_normal(self.mu_1G, self.sigma, size =n_min)
        X_0R  = self.rng.multivariate_normal(self.mu_0R, self.sigma, size =n_min)

        X = np.concatenate((X_1R, X_1G, X_0R, X_0G))
        y = np.concatenate((np.ones(n_maj), np.ones(n_min),np.zeros(n_min), np.zeros(n_maj)))
        group = np.concatenate((np.ones(n_maj), np.zeros(n_min),np.ones(n_min), np.zeros(n_maj)))

        if (as_df):
            data = pd.DataFrame(X)
            data['target'] = y
            data['group'] = group
            return data
        else:
            return X,y,group
    
    def _prob(self,x):
        pos_prob = 2 * ((1/2-self.pi_0) * st.multivariate_normal.pdf(x, self.mu_1R,self.sigma) + (self.pi_0) * st.multivariate_normal.pdf(x,self.mu_1G,self.sigma))
        neg_prob = 2* ((self.pi_0) * st.multivariate_normal.pdf(x, self.mu_0R,self.sigma) + (1/2-self.pi_0) * st.multivariate_normal.pdf(x,self.mu_0G,self.sigma))
        marginal = (pos_prob + neg_prob)/2
        return pos_prob * 0.5/marginal

    def mmse_estimate(self,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(y - (self._prob(X)>1/2),2).mean()

    def model_mmse_estimate(self,w,b,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(X@w +b - self._prob(X),2).mean()

class XORGaussian:
    def __init__(self, delta_D: np.ndarray, delta_C: np.ndarray, sigma: np.ndarray, mu: np.ndarray | float =0, pi_0:float = 1/4, seed:int=42):
        # Given
        self.delta_D = np.array(delta_D)
        self.delta_C = np.array(delta_C)
        self.sigma = np.array(sigma)
        self.mu = np.array(mu)
        self.pi_0 = float(pi_0)
        self.rng = np.random.default_rng(int(seed))

        # Calculated from definition of mu and deltas
        self.mu_1R = self.mu + self.delta_D / 2 - self.delta_C /2
        self.mu_1G = self.mu + self.delta_D / 2 + self.delta_C /2
        self.mu_0R = self.mu - self.delta_D / 2 - self.delta_C /2
        self.mu_0G = self.mu - self.delta_D / 2 + self.delta_C /2
        self.mu_1 = 2* self.pi_0 * (self.mu_1G) +(1-2*pi_0)* (self.mu_1R)
        self.mu_0 = 2* self.pi_0 * (self.mu_0R) +(1-2*pi_0)* (self.mu_0G)
        self.delta_bar = self.mu_1 - self.mu_0

        self._X = None
        self._y = None
        self._group = None
    
    def generate(self, n: int, as_df: bool = True) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_maj = int(n * (1/2 - self.pi_0))
        n_min = int((n - 2*n_maj) // 2)

        X_1R  = self.rng.multivariate_normal(self.mu_1R, self.sigma, size =n_maj)
        X_0G  = self.rng.multivariate_normal(self.mu_0G, self.sigma, size =n_maj)
        X_1G  = self.rng.multivariate_normal(self.mu_1G, self.sigma, size =n_min)
        X_0R  = self.rng.multivariate_normal(self.mu_0R, self.sigma, size =n_min)

        X = np.concatenate((X_1R, X_1G, X_0R, X_0G))
        y = np.concatenate((np.ones(n_maj), np.zeros(n_min),np.zeros(n_min), np.ones(n_maj)))
        group = np.concatenate((np.ones(n_maj), np.zeros(n_min),np.zeros(n_min), np.ones(n_maj)))

        if (as_df):
            data = pd.DataFrame(X)
            data['target'] = y
            data['group'] = group
            return data
        else:
            return X,y,group
    
    def _prob(self,x):
        pos_prob = 2 * ((1/2-self.pi_0) * st.multivariate_normal.pdf(x, self.mu_1R,self.sigma) + (1/2-self.pi_0) * st.multivariate_normal.pdf(x,self.mu_0G,self.sigma))
        neg_prob = 2* ((self.pi_0) * st.multivariate_normal.pdf(x, self.mu_1G,self.sigma) + (self.pi_0) * st.multivariate_normal.pdf(x,self.mu_0R,self.sigma))
        marginal = (1-2*self.pi_0)*pos_prob + (2*self.pi_0)*neg_prob
        return pos_prob * (1-2*self.pi_0)/marginal

    def mmse_estimate(self,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(y - self._prob(X),2).mean()

    def model_mmse_estimate(self,w,b,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(X@w +b - self._prob(X),2).mean()



@staticmethod
def group_balance(data:pd.DataFrame) -> pd.DataFrame:
    groups = data.copy().groupby(['target','group'])
    balanced = groups.apply(lambda df: df.sample(groups.size().min())).reset_index(drop=True)
    return balanced

@staticmethod
def group_sln(data:pd.DataFrame, p:float, seed=42) -> pd.DataFrame:
    noisy = data.copy()
    np.random.seed(seed)
    noisy_indices = np.random.choice(list(range(len(noisy['group']))),size=int(p*len(noisy)), replace=False)
    noisy.loc[noisy_indices, 'group'] = np.abs(noisy.loc[noisy_indices, 'group'] -1)
    return noisy

@staticmethod
def target_sln(data:pd.DataFrame, p:float, seed=42) -> pd.DataFrame:
    noisy = data.copy()
    labels = np.unique(noisy['target'])
    rng = np.random.default_rng(seed=seed)
    noisy_indices = rng.choice(list(range(len(noisy['target']))),size=int(p*len(noisy)), replace=False)
    lab = np.tile(labels.reshape(1,-1), (int(p*len(noisy)),1))
    noisy.loc[noisy_indices, 'target'] = rng.choice(lab[lab!=noisy.loc[noisy_indices, 'target'].to_numpy().reshape(-1,1)].reshape(-1,1), axis=1)
    return noisy

@staticmethod
def group_mixup(data:pd.DataFrame, alpha:float = 1) -> pd.DataFrame:
    n=len(data)
    mixed = pd.DataFrame(columns=data.columns)
    for _, group in data.groupby('target'):
        if (len(group) != 0):
            lam = np.random.beta(alpha,alpha,size=n//2)[:,np.newaxis]
            resampled = []
            for _,sub_group in group.groupby('group'):
                resampled.append(sub_group.sample(n//2,replace=True).to_numpy())
            if len(resampled) <2:
                convex = resampled[0]
                print('Group missing')
            else:
                convex = (resampled[0]* lam) + ((1-lam) * resampled[1])
            mixed = pd.concat([mixed,pd.DataFrame(convex,columns = data.columns) ],ignore_index=True)
        else:
            print('Class missing')
    mixed.group = mixed.group.astype(float)
    mixed.target = mixed.target.astype(int)
    return mixed