import torch, math, random
import numpy as np
import pandas as pd

from data import *
from models import *


def test(
    datasets_list,
    dataset_num,
    checkpoint_path,
    mode,
    device,
    lstmdim=256,
    search_space=1000,
    searchtimes=1, 
    num_iter=10,
    optnum=20, 
    maxepoch=1,
    gamma=0.5,
    pri=True,
    active_flag=True,
    epsilon=0.1,
):
    almodel = torch.load(checkpoint_path, map_location=device)
    hx = torch.randn(1, lstmdim) 
    cx = torch.randn(1, lstmdim) 

    allosses = 0
    alres = 0
    alsotas = 0
    aldises = 0
    alsteps = 0

    # choose the task
    y_obj_no = dataset_num
    dataset = datasets_list

    # random sample 100 molecules
    if len(dataset)<search_space:
        search_space = len(dataset)
        random_choose_X_index = np.array(range(0,search_space))
    else:
        random_choose_X_index = np.random.randint(0, len(dataset), search_space)
    dataset = dataset.select(random_choose_X_index)
    X = dataset.X
    ids = dataset.ids
    if len(dataset.y.shape) == 1:
        y = dataset.y
        task_name = dataset.get_task_names()[0]
    else:
        if dataset.y.shape[1] != 1:
            y = dataset.y[:,y_obj_no]
            task_name = dataset.get_task_names()[y_obj_no]
        else:
            y = dataset.y
            task_name = dataset.get_task_names()[0]
    y = y.reshape(-1, 1)


    #if X.shape[0] != search_space :
    #    raise Exception("X error!!!") 
    
    # Maxima, minima, and starting molecule in the current search space
    GT_max_point, GT_min_point, initial_point = compute_extreme_and_initial_point(X, y, ids, search_space)

    alloss, alxgbmodel, alre, alsota, alstep, identify_flag = run_al_epoch(
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
        active_flag=True,
        epsilon=0.1,
    )
    aldis = math.fabs(alsota - GT_max_point.y.item())/(GT_max_point.y.item() + 1e-5)

    allosses += alloss
    alres += alre
    alsotas += alsota
    aldises += aldis
    alsteps += alstep
    # print("this search reward %.4f" % alre)
    # print("---Ending the search(%d experiments)---" % (num_iter))


    return alres, alsotas, aldises, alsteps, identify_flag
