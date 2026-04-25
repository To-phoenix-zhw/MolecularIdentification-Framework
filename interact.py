#!/usr/bin/env python
# coding: utf-8
import logging
import requests
deepchem_logger = logging.getLogger('deepchem')
deepchem_logger.setLevel(logging.CRITICAL)
dc_logger1= logging.getLogger('dc')
dc_logger1.setLevel(logging.CRITICAL)
import os, torch, math, random, pickle, time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import deepchem as dc
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')



def mksure_path(dirs_or_files):
    if not os.path.exists(dirs_or_files):
        os.makedirs(dirs_or_files)


from optparse import OptionParser
parser = OptionParser()
parser.add_option("--seed", dest="seed", default=0)
parser.add_option("--lstmdim", dest="lstmdim", default=256)
parser.add_option("--search_space", dest="search_space", default=100)
parser.add_option("--searchtimes", dest="searchtimes", default=3)
parser.add_option("--num_iter", dest="num_iter", default=50)
parser.add_option("--optnum", dest="optnum", default=20)
parser.add_option("--maxepoch", dest="maxepoch", default=60000)
parser.add_option("--pri", dest="pri", default='true')
parser.add_option("--active_flag", dest="active_flag", default='true')
parser.add_option("--mode", dest="mode", default='train')
parser.add_option("--custom", dest="custom", default='')
parser.add_option("--case_task", dest="case_task", default=0)
parser.add_option("--data_path", dest="data_path", default='datasets')
parser.add_option("--checkpoint_path", dest="checkpoint_path", default='')
parser.add_option("--model_path", dest="model_path", default='')
parser.add_option("--continue_epoch", dest="continue_epoch", default=0)
parser.add_option("--continue_rewards", dest="continue_rewards", default=0)
parser.add_option("--continue_distances", dest="continue_distances", default=0)
parser.add_option("--test_path", dest="test_path", default='')
parser.add_option("--test_times", dest="test_times", default=1)
opts,args = parser.parse_args()

def set_seed(seed=0):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  

seed = int(opts.seed) 
set_seed(seed)
lstmdim = int(opts.lstmdim)  # 
search_space = int(opts.search_space)  
searchtimes = int(opts.searchtimes)  
num_iter = int(opts.num_iter)  
optnum = int(opts.optnum)  # 
maxepoch = int(opts.maxepoch) #
active_flag = True if (opts.active_flag == 'true') else False 
mode = str(opts.mode)
custom = str(opts.custom)  
case_task = int(opts.case_task)  
data_path = str(opts.data_path) + "/"
checkpoint_path = str(opts.checkpoint_path) 
model_path = str(opts.model_path)  
continue_epoch = int(opts.continue_epoch)
continue_rewards = float(opts.continue_rewards)
continue_distances = float(opts.continue_distances)
test_path = str(opts.test_path)  
test_times = int(opts.test_times)  

mksure_path(data_path)




class Point(object):
    def __init__(self, idx, X, y, smi):
        self.idx = idx
        self.X = X
        self.y = y
        self.smi = smi
    def __str__(self):
        return "{}, {}".format(self.y, self.smi)




class MLP(nn.Module):
    def __init__(self,in_dim, out_dim, hiddensize = 128):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_dim, hiddensize)  # 5*5 from image dimension
        self.fc2 = nn.Linear(hiddensize, out_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = F.relu(self.fc1(x))
        pred = (self.fc2(x))

        if y is not None:
            loss = self.criterion(pred, y)
            return pred, loss
        else:
            return pred



class ActiveModel(nn.Module):
    def __init__(self, lstmdim=256, out_dim=1):
        super(ActiveModel, self).__init__()
        self.dr = MLP(lstmdim*4, lstmdim)
        self.rnn = nn.LSTMCell(lstm+2, lstmdim) # [input_size, hidden_size]
        self.mlp = MLP(lstmdim*2+4, out_dim) 
    
    def forward(
        self, 
        train_X, train_y,
        SOTA, 
        former_X, former_y, 
        hx, cx
    ):
        newX = former_X[-1, :]  
        newy = former_y[-1, :].item() 
        shaped_SOTA = torch.FloatTensor([[SOTA]]).cuda()
        shaped_newX = self.dr(newX.unsqueeze(dim=0).cuda())
        shaped_newy = torch.FloatTensor([[newy]]).cuda()
        rnninput = torch.cat((shaped_SOTA, torch.cat((shaped_newX, shaped_newy), dim=1)), dim=1)  # [1, 1+256+1=258]
        hx, cx = self.rnn(rnninput, (hx, cx)) # [batch, hidden_size]
        
        train_X2 = (train_X ** 2).sum(dim=1).reshape((-1, 1))
        former_X2 = (former_X ** 2).sum(dim=1)
        D = train_X2 + former_X2 - 2 * train_X.mm(former_X.t())
        minD = torch.sqrt(torch.min(D, dim=1).values).reshape((-1, 1))  # [999, 1]

        max_pred = torch.max(train_y)  # torch.max([scale]) 
        min_pred = torch.min(train_y)  # torch.min([scale]) 
        
        prop = (train_y - min_pred) / (max_pred - min_pred + 1e-5)  # [999, 1]  
        
        inputs = torch.cat([hx.repeat(train_X.size(0), 1), # [ready0=999, 256]
                            prop,  # [999, 1]
                            self.dr(train_X), train_y,  # [999, 256], [999, 1]  
                            minD, # [999, 1]
                            train_y - SOTA], 1)  # [999, 1]
        
        pred = self.mlp(inputs).squeeze() # [scale]
        score = nn.Softmax(dim=0)(pred) # [scale]
        
        return score, hx, cx




def choose_experimental_x(size, scores, active_flag=True, epsilon=0.9, mode='train'):
    if not active_flag:
        return random.randint(0, size-1)

    if mode == "train":
        if random.random() <= epsilon:
            return random.randint(0, size-1)
        else:
            return torch.argmax(scores).item() 
    elif mode == "test" or mode == "case" or mode == "custom":
        return torch.argmax(scores).item()  
    else:
        raise Exception("mode error!!!") 




def fitting_model(X, y, num_epoch=1000, early_stopping_rounds=5):
    """
    X: [1000, 1024]
    y: [1000, 1]
    """
    rX = X.cpu().numpy()
    ry = y.cpu().numpy()
    reg = xgb.XGBRegressor(n_estimators=num_epoch, tree_method="gpu_hist", eval_metric=mean_squared_error)
    reg.fit(rX, ry, eval_set=[(rX, ry)], early_stopping_rounds=early_stopping_rounds, verbose=0)
    #print("xgb rmse: %.4f in %d" % (mean_squared_error(ry, reg.predict(rX), squared=False), reg.best_iteration))

    return reg, reg.best_iteration





def run_al_epoch(
    X, y, ids,
    GT_max_point, GT_min_point, 
    initial_point, 
    lstmdim,
    almodel, 
    hx, cx,
    num_iter=10, 
    active_flag=True,
):
    
    loss = 0.0
    search_space = X.shape[0]  
    X_dim = X.shape[1]  # 
    
    already_dataX = (initial_point.X).copy() 
    already_datay = (initial_point.y).copy() 
    already_dataid = np.array([initial_point.smi], dtype=object)  
    # print("already data shape", already_dataX.shape, already_datay.shape, already_dataid.shape)
    
    ready_dataX = X.copy()
    ready_dataX = np.delete(ready_dataX, initial_point.idx, axis=0) 
    ready_datay = y.copy() 
    ready_datay = np.delete(ready_datay, initial_point.idx, axis=0)  
    ready_dataid = ids.copy() 
    ready_dataid = np.delete(ready_dataid, initial_point.idx, axis=0)     
    SOTA = initial_point.y.item()
    logps = []
    rewards = []
    succ_step= 0
    patient = 0
    
    for i in range(num_iter):
        print("-"*8 +f"     Step {i+2}     " + "-"*8)
        datatensorX = torch.FloatTensor(already_dataX).detach().cuda()  
        datatensory = torch.FloatTensor(already_datay).detach().cuda() 
        model, _ = fitting_model(datatensorX, datatensory, num_epoch=1000)
        train_X = torch.FloatTensor(ready_dataX).cuda()  # 
        train_y = torch.FloatTensor(model.predict(ready_dataX)).unsqueeze(dim=1).cuda()          
        scores, hx, cx = almodel(train_X, train_y, SOTA,
                                 datatensorX, datatensory,
                                 hx.cuda(), cx.cuda())
        
        index = choose_experimental_x(train_X.size(0), scores, active_flag=active_flag, mode=mode)
        logp = torch.log(scores[index])  # 
        measuredX = train_X[index]  #   [1024]
        measuredy = ready_datay[index].item()  #  [] 

        print(f'Framework: Selects the Compound: {ready_dataid[index]}')
        print(f'Human:      Conducts wet-lab experiments and returns the objective value: {measuredy:.2f}')
        
        if (measuredy > SOTA) and (math.fabs(measuredy-SOTA)>1e-5):
            reward = (measuredy - SOTA)/(GT_max_point.y.item() - initial_point.y.item() + 1e-5)
            SOTA = measuredy
            succ_step=i
            patient = 0
        else:
            reward = 0
            patient +=1
            if patient>3:
                print(f'Framework: Stops optimization stops since no higher values are obtained in recent 4 steps!')
                break
        
        already_dataX = np.concatenate((already_dataX, ready_dataX[index].reshape(1, -1)))
        already_datay = np.concatenate((already_datay, ready_datay[index].reshape(1, -1)))
        already_dataid = np.concatenate((already_dataid, np.array([ready_dataid[index]], dtype=object)))
        # print("already data shape", already_dataX.shape, already_datay.shape, already_dataid.shape)
        ready_dataX = np.delete(ready_dataX, index, axis = 0)
        ready_datay = np.delete(ready_datay, index, axis = 0)
        ready_dataid = np.delete(ready_dataid, index, axis = 0)
        # print("ready data shape", ready_dataX.shape, ready_datay.shape, ready_dataid.shape)
        logps.append(logp)  
        rewards.append(reward)  
        
    
    return loss, model, np.sum(rewards), SOTA, succ_step


def compute_extreme_and_initial_point_appoint(X, y, ids, initial=0):
    
    X_dim = X.shape[1]  
    maxy_index = np.argmax(y)
    maxy_X = X[maxy_index].reshape(-1, X_dim)  # [1, 1024]
    maxy_y = y[maxy_index].reshape(-1, 1)  # [1, 1]
    maxy_id = ids[maxy_index]  # a string
    GT_max_point = Point(maxy_index, maxy_X, maxy_y, maxy_id)

    miny_index = np.argmin(y)
    miny_X = X[miny_index].reshape(-1, X_dim)
    miny_y = y[miny_index].reshape(-1, 1)
    miny_id = ids[miny_index]
    GT_min_point = Point(miny_index, miny_X, miny_y, miny_id)

    randomindex = initial
    initial_X = X[randomindex].reshape(-1, X_dim)  # [1, 1024]
    initial_y = y[randomindex].reshape(-1, 1)  # [1, 1]
    initial_id = ids[randomindex]  # a string
    initial_point = Point(randomindex, initial_X, initial_y, initial_id)
    print(f'Note: Starting from the compound: {initial_id}')
    
    return GT_max_point, GT_min_point, initial_point 


def case(
    dataset,
    initial=0,
    lstmdim=256,
    search_space=1000,
    searchtimes=1, 
    num_iter=10,
    optnum=20, 
    maxepoch=1,
    active_flag=True,
):
    almodel = torch.load(checkpoint_path)
    hx = torch.randn(1, lstmdim)     
    cx = torch.randn(1, lstmdim) 

    allosses = 0
    alres = 0
    alsotas = 0
    aldises = 0

    X = dataset.X
    ids = dataset.ids
    y = dataset.y.reshape(-1, 1)

    GT_max_point, GT_min_point, initial_point = compute_extreme_and_initial_point_appoint(X, y, ids, initial)

    alloss, alxgbmodel, alre, alsota, succ_step = run_al_epoch(
        X, y, ids,
        GT_max_point, GT_min_point, 
        initial_point, 
        lstmdim,
        almodel, 
        hx, cx,
        num_iter,
        active_flag=True,
    )
    aldis = math.fabs(alsota - GT_max_point.y.item())/(GT_max_point.y.item() + 1e-5)

    allosses += alloss
    alres += alre
    alsotas += alsota
    aldises += aldis


    return alres, alsotas, aldises, succ_step







if __name__ == "__main__":

    steps_succ=[0 for i in range(10)]
    dataset = dc.data.DiskDataset(custom)
    print("*"*8 + "Lead Optimization Process of Framework" + "*"*8)
    alre, alsota, aldis, succ_step = case(
        dataset,
        0,  
        lstmdim,
        search_space,
        searchtimes, 
        num_iter,
        optnum,
        maxepoch,
        active_flag
    )
    if aldis < 1e-5:
        print("*"*8 + "                     End                 " + "*"*8)
        print("*"*8 + f"Identified the Optimal Compound at Step {succ_step+2}" + "*"*8)
    else:
        print("*"*8 + "                        End                       " + "*"*8)
        print("*"*8 + f"Failed to Identify the Optimal Compound in {num_iter} Steps" + "*"*8)



