import torch
import torch.nn as nn
import numpy as np
from typing import Literal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class OptLinear:
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, x, theta):
        a = np.cov(x,theta.reshape(-1,1),rowvar=False,ddof=1)[-1][:-1]
        self.a = a @ np.linalg.inv(np.cov(x,rowvar=False,ddof=1))
        self.b = theta.mean() - a.T@ x.mean(axis=0)
        return self

    def predict_proba(self, x):
        return 1/(1+np.exp(-x@self.a - self.b))
        
class NeuralNetwork:
    def __init__(self, loss:Literal['log','square']='log',device:Literal['cpu','mps','cuda']|None = None) -> None:
        self.model = None
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        if device is not None:
            self.device = device
        match loss :
            case 'log':
                self.loss = nn.CrossEntropyLoss().to(self.device)
            case 'square':
                self.loss = nn.MSELoss().to(self.device)
            case _:
                raise ValueError('Unknown loss')

    def fit(self,X,y,opt:Literal['SGD','AdamW']|None = 'SGD', lr: float|None = 0.001, scheduler:Literal['cosine','linear']|None = None, batch_size: int = 64, epochs:int = 100,verbose=False,n_hidden=100) -> None:
        '''Run Gradient Descent'''
        if self.model is None:
            self.model = nn.Sequential(nn.Linear(X.shape[1],n_hidden),nn.ReLU(), nn.Linear(n_hidden,1),nn.Flatten(),nn.Sigmoid(),nn.Flatten(0)).to(self.device)
        match opt:
            case 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
            case 'AdamW':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            case _:
                self.optimizer = None
        match scheduler:    
            case 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
            case 'linear':
                self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)
            case _:
                self.scheduler = None
        x_tensor = torch.tensor(X,device=self.device,dtype=torch.float32)
        y_tensor = torch.tensor(y,device=self.device,dtype=torch.float32)
        counter = (lambda x: x) if verbose == False else (lambda x: x)
        for epoch in counter(range(epochs)):
            #x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x_tensor)
            loss_output = self.loss(output, y_tensor)
            loss_output.backward()
            if verbose:
                # print(x_tensor,output,y_tensor)
                print(loss_output.detach())
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
        return self
    
    def predict_proba(self,X):
        '''Return soft prediction'''
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.model(X).detach().cpu().numpy()
        
    def predict(self,X):
        '''Return hard prediction'''
        return (self.predict_proba(X) > 1/2).astype(int)

class LogisticRegression:
    def __init__(self, loss:Literal['log','square']='log',device:Literal['cpu','mps','cuda']|None = None) -> None:
        self.model = None
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        if device is not None:
            self.device = device
        match loss :
            case 'log':
                self.loss = nn.CrossEntropyLoss().to(self.device)
            case 'square':
                self.loss = nn.MSELoss().to(self.device)
            case _:
                raise ValueError('Unknown loss')

    def initialize_model(self,d):
        self.model = nn.Sequential(nn.Linear(d,1),nn.Flatten(),nn.Sigmoid(),nn.Flatten(0)).to(self.device)

    def fit(self,X,y,opt:Literal['SGD','AdamW']|None = 'SGD', lr: float|None = 0.001, scheduler:Literal['cosine','linear']|None = None, batch_size: int = 64, epochs:int = 100,verbose=False) -> None:
        '''Run Gradient Descent'''
        if self.model is None:
            self.initialize_model(X.shape[1])
        match opt:
            case 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
            case 'AdamW':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            case _:
                self.optimizer = None
        match scheduler:    
            case 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
            case 'linear':
                self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)
            case _:
                self.scheduler = None
        x_tensor = torch.tensor(X,device=self.device,dtype=torch.float32)
        y_tensor = torch.tensor(y,device=self.device,dtype=torch.float32)
        counter = (lambda x: x) if verbose == False else (lambda x: x)
        for epoch in counter(range(epochs)):
            #x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x_tensor)
            loss_output = self.loss(output, y_tensor)
            loss_output.backward()
            if verbose:
                print(x_tensor,output,y_tensor)
                print(loss_output.detach())
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
        return self
    
    def predict_proba(self,X):
        '''Return soft prediction'''
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.model(X).detach().cpu().numpy()
        
    def predict(self,X):
        '''Return hard prediction'''
        return (self.predict_proba(X) > 1/2).astype(int)
