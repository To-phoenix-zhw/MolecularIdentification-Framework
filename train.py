import torch, math, random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from models import *
from data import *
from utils import *

def train(
    datasets_list,
    val_datasets_list,
    save_path,
    model_path,
    continue_epoch,
    continue_rewards,
    continue_distances,
    mode,
    logger,
    device,
    lstmdim=256,
    search_space=50,
    searchtimes=3, 
    num_iter=10,
    optnum=20, 
    maxepoch=20000,
    gamma=0.5,
    pri=False,
    active_flag=True,
    epsilon=0.1,
):
    rewards = 0  
    distances = 0  
    losses    = 0
    rewards_list = [] 
    models_list = [] 
    SOTAs_list = []  

    logger.info('Building model...')
    logscale = 1000
    checkpoint_path = save_path

    almodel = ActiveModel(lstmdim).to(device)
    if model_path != '':
        almodel = torch.load(model_path)
        # print("Load Model: %s, continue training." % (model_path))
        rewards = continue_rewards*continue_epoch
        distances = continue_distances*continue_epoch
    
    aloptimizer = optim.Adam(almodel.parameters(), lr=1e-4)  
    almodel.train() 
    
    train_nos = list(range(len(datasets_list)))
    #train_loop = tqdm(range(maxepoch), desc='Training')
    train_loop = range(maxepoch)
    previous_val_r= 0

    logger.info('Training model...')
    for cur_epo in train_loop:
        # random choose a task
        dataset_obj_idx = np.random.choice(train_nos)
        dataset = datasets_list[dataset_obj_idx]

        # random sample 100 molecules
        random_choose_X_index = np.random.randint(0, len(dataset), search_space)
        dataset = dataset.select(random_choose_X_index)
        X = dataset.X
        ids = dataset.ids
        y = dataset.y.reshape(-1, 1)
        task_name = dataset.get_task_names()[0]

        if X.shape[0] != search_space:
            raise Exception("X error!!!") 
            
        # print("*****EPOCH %d TASK %d %s *****" % (cur_epo+1, dataset_obj_idx, task_name)) 
        # print(X.shape, y.shape, ids.shape)  

        if cur_epo < continue_epoch:
            continue

        reward_list = []  
        loss_list = [] 
        model_list = [] 
        SOTA_list = [] 
        
        # Maxima, minima, and starting molecule in the current search space
        GT_max_point, GT_min_point, initial_point = compute_extreme_and_initial_point(X, y, ids, search_space)
    
        # search
        for j in range(searchtimes):
            # print("---Into the search " + str(j+1) +  " (%d experiments)---" % num_iter)

            hx = torch.randn(1, lstmdim) 
            cx = torch.randn(1, lstmdim) 
            loss, model, re, sota, step,_ = run_al_epoch(
                X, y, ids,
                GT_max_point, GT_min_point, 
                initial_point, 
                lstmdim,
                almodel, 
                hx, cx,
                mode,
                device,
                gamma,
                num_iter,
                pri, 
                active_flag,
                epsilon,
            )
            loss_list.append(loss)
            reward_list.append(re)
            model_list.append(model)
            SOTA_list.append(sota)
            # print("this search reward %.4f" % re)
            # print("---Ending the search(10 experiments)---")
        
        # Gradients: only use losses of search processes whose reward is bigger than the average 
        cur_re = 0
        cnt_re = 0
        cur_dis = 0
        cur_loss = 0.0
        for id, re in enumerate(reward_list):
            if (re > np.mean(reward_list)) or (math.fabs(re-np.mean(reward_list))<1e-5):  
                (loss_list[id] / optnum).backward()  
                cur_re += re
                cnt_re += 1
                cur_dis += (math.fabs(SOTA_list[id] - GT_max_point.y.item())/(GT_max_point.y.item() + 1e-5))
                cur_loss += loss_list[id].item()

        # Update parameters every 20 episodes
        if cur_epo % optnum == optnum - 1: 
            clip_grad_norm_(almodel.parameters(), 5.0)  
            aloptimizer.step()  
            aloptimizer.zero_grad() 
            almodel.zero_grad()  
        
        rewards += (cur_re/cnt_re)
        distances += (cur_dis/cnt_re)
        rewards_list.append(rewards/(cur_epo+1))
        models_list.append(model_list[np.argmax(reward_list)])
        SOTAs_list.append(SOTA_list[np.argmax(SOTA_list)])
        losses += cur_loss/cnt_re

        lognum = 40*optnum

        if ((cur_epo)>0) and ((cur_epo)%lognum==0):
            if ((cur_epo)<=logscale) : rewards = rewards*(1-optnum*1.0/lognum)
            logger.info('[Training] Iter %d | reward %.3f |  loss %.3f' % (cur_epo, rewards/lognum, losses/lognum))
            rewards = 0
            losses = 0


        if ((cur_epo)>0) and ((cur_epo)%(logscale*optnum)==0):
            val_epoch = 100
            val_rewards = validation(almodel,val_datasets_list,logger,device,lstmdim,search_space,searchtimes, num_iter, optnum, val_epoch, gamma, pri, active_flag, epsilon)

            logger.info('[Validation] Iter %d | reward %.6f' % (cur_epo, val_rewards))
            if val_rewards>previous_val_r:
                torch.save(almodel, checkpoint_path + '/almodel_'+ str(cur_epo) + '.pt')
                logger.info('[Validation] New Best model!')
                previous_val_r=val_rewards
            almodel.train()
 
        
        #train_loop.set_description(f'Train Iter [{cur_epo+1}/{maxepoch}]')
        #train_loop.set_postfix(reward = rewards/(cur_epo+1))
        # print("*****ENDING THE EPOCH %d TASK %d %s *****" % (cur_epo+1, dataset_obj_idx, task_name)) 

    return rewards_list



def validation(
    almodel,
    val_datasets_list,
    logger,
    device,
    lstmdim=256,
    search_space=50,
    searchtimes=3, 
    num_iter=10,
    optnum=20, 
    maxepoch=1000,
    gamma=0.5,
    pri=False,
    active_flag=True,
    epsilon=0.1,
):
    rewards = 0  
    distances = 0  
    rewards_list = [] 
    models_list = [] 
    SOTAs_list = []  

    almodel.eval() 
    hx = torch.randn(1, lstmdim) 
    cx = torch.randn(1, lstmdim) 
    
    indexes = list(range(len(val_datasets_list)))
    val_loop = tqdm(range(maxepoch), desc='Validation')

    for cur_epo in val_loop:
        # random choose a task
        dataset_obj_idx = np.random.choice(indexes)
        dataset = val_datasets_list[dataset_obj_idx]

        # random sample 100 molecules
        random_choose_X_index = np.random.randint(0, len(dataset), search_space)
        dataset = dataset.select(random_choose_X_index)
        X = dataset.X
        ids = dataset.ids
        y = dataset.y.reshape(-1, 1)
        task_name = dataset.get_task_names()[0]

        if X.shape[0] != search_space:
            raise Exception("X error!!!") 
            
        reward_list = []  
        loss_list = [] 
        
        GT_max_point, GT_min_point, initial_point = compute_extreme_and_initial_point(X, y, ids, search_space)
    
        # search
        mode = 'test'
        loss, model, re, sota, step,_ = run_al_epoch(
                X, y, ids,
                GT_max_point, GT_min_point, 
                initial_point, 
                lstmdim,
                almodel, 
                hx, cx,
                mode,
                device,
                gamma,
                num_iter,
                pri, 
                active_flag,
                epsilon,
            )
       
        rewards_list.append(re)
        val_loop.set_description(f'Val iter [{cur_epo+1}/{maxepoch}]')
        val_loop.set_postfix(reward = np.mean(rewards_list))

    avg = np.mean(rewards_list)

    return avg
