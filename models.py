import torch
import torch.nn as nn
import numpy as np
from typing import Literal
from torch.utils.data import DataLoader, TensorDataset

class LogisticRegression:
    def __init__(self, loss:Literal['log','square']='log',device:Literal['cpu','mps','cuda']|None = None) -> None:
        match loss :
            case 'log':
                self.loss = nn.CrossEntropyLoss()
            case 'square':
                self.loss = nn.MSELoss()
            case _:
                raise ValueError('Unknown loss')
        self.model = None
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        if device is not None:
            self.device = device

    def fit(self,X,y,opt:Literal['SGD','AdamW']|None = 'SGD', lr: float|None = 0.001, scheduler:Literal['cosine','linear']|None = None, batch_size: int = 64, epochs:int = 100) -> None:
        '''Run Gradient Descent'''
        if self.model is None:
            self.model = nn.Sequential(nn.Linear(X.shape[1],1),nn.Flatten(),nn.Sigmoid(),nn.Flatten(0)).to(self.device)
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
        dataset = TensorDataset(torch.tensor(X,device=self.device,dtype=torch.float32), torch.tensor(y,device=self.device,dtype=torch.float32))
        trainloader = DataLoader(dataset,batch_size=batch_size)
        for epoch in range(epochs):
            for i, (x, y) in enumerate(trainloader):
                x, y, = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss_output = self.loss(output, y)
                loss_output.backward()
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