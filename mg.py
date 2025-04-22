import numpy as np
import pandas as pd
import scipy.stats as st

class CCGaussian:
    def __init__(self, mu_0: np.ndarray, mu_1: np.ndarray, sigma_0: np.ndarray,sigma_1: np.ndarray, sigma: float, p:float = 1/2, seed:int=42):
        # Given
        self.mu_0 = np.array(mu_0)
        self.mu_1 = np.array(mu_1)
        self.sigma_0 = np.array(sigma_0)
        self.sigma_1 = np.array(sigma_1)
        self.sigma = float(sigma)
        self.p = float(p)
        self.rng = np.random.default_rng(int(seed))

        self._X = None
        self._y = None
        self._group = None
    
    def generate(self, n: int, as_df: bool = True) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(n)
        y = self.rng.choice([0,1],size = n, p=[1-self.p,self.p])
        X_0 = self.rng.multivariate_normal(mean = self.mu_0, cov = self.sigma_0, size = n)
        X_1 = self.rng.multivariate_normal(mean = self.mu_1, cov = self.sigma_1, size = n)
        X = np.where((y==1).reshape(-1,1),X_1,X_0)
        X_sigma = X + self.sigma * self.rng.multivariate_normal(np.zeros_like(self.mu_0), np.eye(self.sigma_1.shape[0]),size=n)

        if (as_df):
            data = pd.DataFrame(X_sigma)
            data['target'] = y
            return data
        else:
            return X_sigma,y
    
    def _prob(self,x):
        pos_prob = st.multivariate_normal.pdf(x, self.mu_1,self.sigma_1 + np.eye(self.sigma_1.shape[0]) * self.sigma**2) 
        neg_prob = st.multivariate_normal.pdf(x, self.mu_0,self.sigma_0 + np.eye(self.sigma_0.shape[0]) * self.sigma**2) 
        marginal = (self.p * pos_prob + (1-self.p) * neg_prob)
        return pos_prob * self.p/marginal

    def _theta(self,x):
        pos_prob = st.multivariate_normal.pdf(x, self.mu_1,self.sigma_1 + np.eye(self.sigma_1.shape[0]) * self.sigma**2) 
        neg_prob = st.multivariate_normal.pdf(x, self.mu_0,self.sigma_0 + np.eye(self.sigma_0.shape[0]) * self.sigma**2)
        return np.log(
            self.p * pos_prob/(
                (1-self.p) * neg_prob
            )
        )

    def mmse_estimate(self,n):
        X, y = self.generate(n,as_df=False)
        return np.power(y - self._prob(X),2).mean()

    def model_mmse_estimate(self,w,b,n):
        X, y = self.generate(n,as_df=False)
        return np.power(X@w +b - self._prob(X),2).mean()

class MixtureGaussian:
    def __init__(self, delta_D: np.ndarray, delta_C: np.ndarray, sigma: np.ndarray, mu: np.ndarray | float =0, pi_0:float = 1/4, sigma_noise:float = 0, seed:int=42):
        # Given
        self.delta_D = np.array(delta_D)
        self.delta_C = np.array(delta_C)
        self.sigma = np.array(sigma)
        self.sigma_noise = float(sigma_noise)
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
        tilde_X = X + self.sigma_noise * self.rng.multivariate_normal(np.zeros_like(self.delta_D), np.eye(len(self.delta_D)), size=len(X))
        y = np.concatenate((np.ones(n_maj), np.ones(n_min),np.zeros(n_min), np.zeros(n_maj)))
        group = np.concatenate((np.ones(n_maj), np.zeros(n_min),np.ones(n_min), np.zeros(n_maj)))

        if (as_df):
            data = pd.DataFrame(tilde_X)
            data['target'] = y
            data['group'] = group
            return data
        else:
            return tilde_X,y,group
    
    def _prob(self,x):
        pos_prob = 2 * (
            (1/2-self.pi_0) * st.multivariate_normal.pdf(
                x, self.mu_1R,self.sigma + self.sigma_noise**2 * np.eye(self.sigma.shape[0])
            ) + (self.pi_0) * st.multivariate_normal.pdf(
                x,self.mu_1G,self.sigma + self.sigma_noise**2 * np.eye(self.sigma.shape[0])
            )
        )
        neg_prob = 2* (
            (self.pi_0) * st.multivariate_normal.pdf(
                x, self.mu_0R,self.sigma + self.sigma_noise**2 * np.eye(self.sigma.shape[0])
            ) + (1/2-self.pi_0) * st.multivariate_normal.pdf(
                x,self.mu_0G,self.sigma + self.sigma_noise**2 * np.eye(self.sigma.shape[0])
            )
        )
        marginal = (pos_prob + neg_prob)/2
        return pos_prob * 0.5/marginal

    def mmse_estimate(self,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(y - self._prob(X),2).mean()

    def model_mmse_estimate(self,w,b,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(X@w +b - self._prob(X),2).mean()

class RandomMixtureGaussian:
    def __init__(self, n_modes_per_class: int, class_sep: float = 2, mode_deviation:float = 2,d:int = None, sigma_noise:float = 0, seed:int=42):
        # Given
        self.d = int(d)
        self.sigma_noise = float(sigma_noise)
        self.n_modes_per_class = int(n_modes_per_class)
        
        self.rng = np.random.default_rng(int(seed))
        self.mu_pos = np.array([class_sep/2/np.sqrt(self.d)]*self.d)
        self.pos_means = self.rng.multivariate_normal(self.mu_pos, mode_deviation**2/self.d * np.eye(self.d), size=n_modes_per_class)
        self.neg_means = self.rng.multivariate_normal(-self.mu_pos, mode_deviation**2/self.d * np.eye(self.d), size=n_modes_per_class)
        self.means = np.stack((self.pos_means.T,self.neg_means.T),axis=2).T
        self.sigma = np.eye(self.d)

        self._X = None
        self._y = None
        self._domain = None
    
    def generate(self, n: int, as_df: bool = True) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(n)
        self._y = self.rng.choice([0,1], replace=True, size = n).astype(int)
        self._domain = self.rng.choice(len(self.pos_means), replace=True, size=n).astype(int)
        self._X = self.rng.multivariate_normal(np.zeros(self.d), self.sigma, size=n) + self.means[self._y,self._domain]
        X_sigma = self._X + self.sigma_noise * self.rng.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=n)

        if (as_df):
            data = pd.DataFrame(X_sigma)
            data['target'] = self._y
            data['domain'] = self._domain
            return data
        else:
            return X_sigma,self._y,self._domain
    
    def _prob(self,x):
        """
        Probability that x in is class 1
        """
        pos_prob = np.array([
            st.multivariate_normal.pdf(
                x, self.means[1,m],self.sigma+ self.sigma_noise**2 * np.eye(self.d)
            )
        for m in range(self.n_modes_per_class)]).mean(axis=0)
        neg_prob = np.array([
            st.multivariate_normal.pdf(
                x, self.means[0,m],self.sigma+ self.sigma_noise**2 * np.eye(self.d)
            )
        for m in range(self.n_modes_per_class)]).mean(axis=0)
        marginal = (pos_prob + neg_prob)/2
        return pos_prob * 0.5/marginal

    def _theta(self,x):
        """
        Optimal soft predictor
        """
        pos_prob = np.array([
            st.multivariate_normal.pdf(
                x, self.means[1,m],self.sigma+ self.sigma_noise**2 * np.eye(self.d)
            )
        for m in range(self.n_modes_per_class)]).mean(axis=0)
        neg_prob = np.array([
            st.multivariate_normal.pdf(
                x, self.means[0,m],self.sigma+ self.sigma_noise**2 * np.eye(self.d)
            )
        for m in range(self.n_modes_per_class)]).mean(axis=0)
        return np.log(pos_prob/neg_prob)

    def mmse_estimate(self,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(y - self._prob(X),2).mean()

    def model_mmse_estimate(self,w,b,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(X@w +b - self._prob(X),2).mean()

class ScalingXORGaussian2D:
    def __init__(self, n_modes_per_class: int, scale:float=1, sigma_noise:float = 0, seed:int=42):
        # Given
        self.sigma_noise = float(sigma_noise)
        self.n_modes_per_class = int(n_modes_per_class)
        self.d = 2
        
        self.rng = np.random.default_rng(int(seed))
        self.means = [[],[]]

        angles = np.linspace(0, 2 * np.pi, n_modes_per_class, endpoint=False)
        delta = angles[1] - angles[0]
        for angle in angles:
            x,y = np.cos(angle),np.sin(angle)
            self.means[0].append([x,y])
            x_alt,y_alt = np.cos(angle+delta/2),np.sin(angle+delta/2)
            self.means[1].append([x_alt,y_alt])
        self.means = np.array(self.means)*scale
        self.sigma = np.eye(self.d)/n_modes_per_class**2

        self._X = None
        self._y = None
        self._domain = None
    
    def generate(self, n: int, as_df: bool = True) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(n)
        self._y = self.rng.choice([0,1], replace=True, size = n).astype(int)
        self._domain = self.rng.choice(self.n_modes_per_class, replace=True, size=n).astype(int)
        self._X = self.rng.multivariate_normal(np.zeros(self.d), self.sigma, size=n) + self.means[self._y,self._domain]
        X_sigma = self._X + self.sigma_noise /self.n_modes_per_class * self.rng.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=n)

        if (as_df):
            data = pd.DataFrame(X_sigma)
            data['target'] = self._y
            data['domain'] = self._domain
            return data
        else:
            return X_sigma,self._y,self._domain
    
    def _prob(self,x):
        """
        Probability that x in is class 1
        """
        pos_prob = np.array([
            st.multivariate_normal.pdf(
                x, self.means[1,m],self.sigma+ self.sigma_noise**2/self.n_modes_per_class**2 * np.eye(self.d)
            )
        for m in range(self.n_modes_per_class)]).mean(axis=0)
        neg_prob = np.array([
            st.multivariate_normal.pdf(
                x, self.means[0,m],self.sigma+ self.sigma_noise**2/self.n_modes_per_class**2 * np.eye(self.d)
            )
        for m in range(self.n_modes_per_class)]).mean(axis=0)
        marginal = (pos_prob + neg_prob)/2
        return pos_prob * 0.5/marginal

    def mmse_estimate(self,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(y - self._prob(X),2).mean()

    def model_mmse_estimate(self,w,b,n):
        X, y, group = self.generate(n,as_df=False)
        return np.power(X@w +b - self._prob(X),2).mean()
class BSCGaussian:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, p:float = 1/4,p_n:float=0.1, seed:int=42):
        # Given
        self.sigma = np.array(sigma)
        self.mu = np.array(mu)
        self.p = float(p)
        self.p_n = float(p_n)
        self.rng = np.random.default_rng(int(seed))

        # Calculated from definition of mu and deltas
        self.mu_1 = 1
        self.mu_0 = 0

        self._X = None
        self._y = None
    
    def generate(self, n: int, as_df: bool = True) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(n)
        y = self.rng.choice([0,1],size = n, p=[1-self.p,self.p])
        X = (self.rng.choice([0,1],size = n, p=[1-self.p_n,self.p_n]) + y)%2
        X_sigma = X + self.sigma * self.rng.standard_normal(size=n)

        if (as_df):
            data = pd.DataFrame(X_sigma)
            data['target'] = y
            return data
        else:
            return X_sigma,y
    
    def _prob(self,x):
        pos_prob =  (1-self.p_n) * st.norm.pdf(x, self.mu_1,self.sigma) +  self.p_n *st.norm.pdf(x,self.mu_0, self.sigma)
        neg_prob = (1-self.p_n) * st.norm.pdf(x, self.mu_0,self.sigma) +  self.p_n *st.norm.pdf(x,self.mu_1, self.sigma)
        marginal = (self.p)*pos_prob + (1-self.p)*neg_prob
        return pos_prob * (self.p)/marginal

    def _theta(self,x):
        pos_prob =  (1-self.p_n) * st.norm.pdf(x, self.mu_1,self.sigma) +  self.p_n *st.norm.pdf(x,self.mu_0, self.sigma)
        neg_prob = (1-self.p_n) * st.norm.pdf(x, self.mu_0,self.sigma) +  self.p_n *st.norm.pdf(x,self.mu_1, self.sigma)
        return np.log(
            self.p * pos_prob/(
                (1-self.p) * neg_prob
            )
        )

    def mmse_estimate(self,n):
        X, y = self.generate(n,as_df=False)
        return np.power(y - (self._prob(X)),2).mean()

    def model_mmse_estimate(self,w,b,n):
        X, y = self.generate(n,as_df=False)
        return np.power(X@w +b - self._prob(X),2).mean()

class BACGaussian:
    def __init__(self, sep: np.ndarray, sigma: np.ndarray, mu: np.ndarray | float =0, pi_0:float = 1/4,pi_1:float=1/4, seed:int=42):
        # Given
        self.sep = np.array(sep)
        self.sigma = np.array(sigma)
        self.mu = np.array(mu)
        self.pi_0 = float(pi_0)
        self.pi_1 = float(pi_1)
        self.rng = np.random.default_rng(int(seed))

        # Calculated from definition of mu and deltas
        self.mu_1R = self.mu - self.sep/2
        self.mu_1G = self.mu + self.sep/2
        self.mu_0R = self.mu - self.sep/2
        self.mu_0G = self.mu + self.sep/2
        self.mu_1 = 2* self.pi_0 * (self.mu_1G) +(1-2*pi_0)* (self.mu_1R)
        self.mu_0 = 2* self.pi_0 * (self.mu_0R) +(1-2*pi_0)* (self.mu_0G)
        self.delta_bar = self.mu_1 - self.mu_0

        self._X = None
        self._y = None
        self._group = None
    
    def generate(self, n: int, as_df: bool = True) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_maj_0 = int(n * (1/2 - self.pi_0))
        n_min_0 = int((n - 2*n_maj_0) // 2)
        n_maj_1 = int(n * (1/2 - self.pi_1))
        n_min_1 = int((n - 2*n_maj_1) // 2)

        X_1R  = self.rng.multivariate_normal(self.mu_1R, self.sigma, size =n_maj_1)
        X_0G  = self.rng.multivariate_normal(self.mu_0G, self.sigma, size =n_maj_0)
        X_1G  = self.rng.multivariate_normal(self.mu_1G, self.sigma, size =n_min_1)
        X_0R  = self.rng.multivariate_normal(self.mu_0R, self.sigma, size =n_min_0)

        X = np.concatenate((X_1R, X_1G, X_0R, X_0G))
        y = np.concatenate((np.ones(n_maj_1), np.ones(n_min_1),np.zeros(n_min_0), np.zeros(n_maj_0)))
        group = np.concatenate((np.ones(n_maj_1), np.zeros(n_min_1),np.ones(n_min_0), np.zeros(n_maj_0)))

        if (as_df):
            data = pd.DataFrame(X)
            data['target'] = y
            data['group'] = group
            return data
        else:
            return X,y,group
    
    def _prob(self,x):
        pos_prob = 2 * ((1/2-self.pi_1) * st.multivariate_normal.pdf(x, self.mu_1R,self.sigma) + (self.pi_1) * st.multivariate_normal.pdf(x,self.mu_1G,self.sigma))
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
