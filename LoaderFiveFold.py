#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import deepchem as dc
import random, pickle
import numpy as np


# In[2]:


ecfp_dataset_path = "/home/zhanghanwen/ACLearning/datasets/ECFP"
data_path = "/home/zhanghanwen/ACLearning/datasets/"


# In[3]:


featurizer='ECFP'
fold = 0
k_fold = 5
seed = 0 


# In[4]:


def set_seed(seed=0):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    print("set seed %d" % seed)


# 设定随机种子
set_seed(seed)


# In[5]:


def dataset_describe(datasets_list):
    datasets_num = len(datasets_list)
    molecules_scale = 0
    tasks_scale = 0
    for dataset in datasets_list:
        molecules_scale += dataset.X.shape[0]
        tasks_scale += dataset.y.shape[1]
    
    print("datasets num: ", datasets_num)
    print("datasets molecules scale: ", molecules_scale)
    print("datasets tasks scale: ", tasks_scale)
    print("*"*8)
    
    return datasets_num, molecules_scale, tasks_scale


# In[6]:


def Quantum_Mechanical_Datasets(featurizer, dataset_path):
    # qm7 1
    qm7tasks, qm7dataset, qm7transformers = dc.molnet.load_qm7(
        featurizer=featurizer, splitter=None, 
        data_dir=dataset_path, save_dir=dataset_path
    )
    qm7dataset = qm7dataset[0]
    

    # qm8 16
    qm8tasks, qm8dataset, qm8transformers = dc.molnet.load_qm8(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    qm8dataset = qm8dataset[0]

    # qm9 12
    qm9tasks, qm9dataset, qm9transformers = dc.molnet.load_qm9(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    qm9dataset = qm9dataset[0]
    
    quantum_datasets_list = [qm7dataset, qm8dataset, qm9dataset]
        
    print("Quantum_Mechanical_Datasets: ")
    dataset_describe(quantum_datasets_list)
    
    return quantum_datasets_list


# In[7]:


def Physical_Chemistry_Datasets(featurizer, dataset_path):
    

    # ESOL 1
    ESOLtasks, ESOLdataset, ESOLtransformers = dc.molnet.load_delaney(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    ESOLdataset = ESOLdataset[0]

    # FreeSolv 1
    FreeSolvtasks, FreeSolvdataset, FreeSolvtransformers = dc.molnet.load_freesolv(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    FreeSolvdataset = FreeSolvdataset[0]

    # Lipophilicity 1
    Lipotasks, Lipodataset, Lipotransformers = dc.molnet.load_lipo(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    Lipodataset = Lipodataset[0] 

    # Thermosol 1
    thermosoltasks, thermosoldataset, thermosoltransformers = dc.molnet.load_thermosol(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    thermosoldataset = thermosoldataset[0]
    
    # HPPB 1
    hppbtasks, hppbdataset, hppbtransformers = dc.molnet.load_hppb(
        featurizer=featurizer, splitter=None, 
        data_dir=dataset_path, save_dir=dataset_path
    )
    hppbdataset = hppbdataset[0]
    
    # HOPV 8
    hopvtasks, hopvdataset, hopvtransformers = dc.molnet.load_hopv(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    hopvdataset = hopvdataset[0]
    
    pc_datasets_list = [ESOLdataset, FreeSolvdataset, Lipodataset, thermosoldataset, hppbdataset, hopvdataset]
    
    print("Physical_Chemistry_Datasets: ")
    dataset_describe(pc_datasets_list)
    
    return pc_datasets_list


# In[9]:


def BioDatasets(featurizer, dataset_path):
    
    # BACE 1
    bacetasks, bacedataset, bacetransformers = dc.molnet.load_bace_regression(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    bacedataset = bacedataset[0]
    
    # PPB 1
    ppbtasks, ppbdataset, ppbtransformers = dc.molnet.load_ppb(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    ppbdataset = ppbdataset[0]
    
    bio_datasets_list = [bacedataset, ppbdataset]
    
    print("BioDatasets: ")
    dataset_describe(bio_datasets_list)
    
    return bio_datasets_list


# In[10]:


def Physiology_Datasets(featurizer, dataset_path):
    # Clearance 1
    clearancetasks, clearancedataset, clearancetransformers = dc.molnet.load_clearance(
        featurizer=featurizer, splitter=None,
        data_dir=dataset_path, save_dir=dataset_path
    )
    clearancedataset = clearancedataset[0]
    
    physi_datasets_list = [clearancedataset]
    
    print("Physiology_Datasets")
    dataset_describe(physi_datasets_list)
    
    return physi_datasets_list


# In[11]:


def AllDatasetsLoader(featurizer='ECFP', dataset_path=ecfp_dataset_path):
    # 分大类加载数据集
    quantum_datasets_list = Quantum_Mechanical_Datasets(featurizer, dataset_path)
    pc_datasets_list = Physical_Chemistry_Datasets(featurizer, dataset_path)
    bio_datasets_list = BioDatasets(featurizer, dataset_path)
    physi_datasets_list = Physiology_Datasets(featurizer, dataset_path)
    
    # 合并数据集
    all_datasets_list = quantum_datasets_list + pc_datasets_list + bio_datasets_list + physi_datasets_list
    print("All_Datasets:")
    all_datasets_num, all_molecules_scale, all_tasks_scale = dataset_describe(all_datasets_list)
    
    return all_datasets_list, all_tasks_scale


# In[14]:


# cnt_all_ids = 0
# for dataset in all_datasets_list:
#     cnt_all_ids += len(dataset.ids)
# cnt_all_ids


# In[15]:


def get_vals_no(a,n):
    # 每个依次当过验证集
    val_no = []
    random.shuffle(a)
    p = True
    while p:
        b=random.sample(a,n)
        b.sort()   #排序
        val_no.append(b)
        print(b)
        a=list(set(a).difference(set(b)))  #去除已抽样的数据
        if len(a) > 0:
            p=True
        else:
            p=False
    
    return val_no


# In[16]:


def get_trains_no(a, vals_no):
    trains_no = []
    for val_no in vals_no:
        b = list(set(a).difference(set(val_no)))
        print(b)
        trains_no.append(b)
    return trains_no


# In[24]:


def get_train_val_fold(tasks_num, fold=0, k_fold=5):
    # 5折交叉验证，5组，随机9等分任务
    tasks_no = list(range(tasks_num))
    elements_num = int(tasks_num / k_fold)
    vals_no = get_vals_no(tasks_no, elements_num)
    trains_no = get_trains_no(tasks_no, vals_no)
    
    # 当前需要生成第几组
    
    return vals_no[fold], trains_no[fold] 



def DatasetsSmiles(all_datasets_list, data_path=data_path):
    for idx, datasets in enumerate(all_datasets_list):
        print(len(datasets.ids), datasets.X.shape[0])
        print(type(datasets.ids))
        smipath = data_path + str(idx) + "smiles.txt"
        if os.path.isfile(smipath):
            continue
        else:
            np.savetxt(smipath, datasets.ids, fmt='%s')
            # np.loadtxt("test.txt", dtype='str')

# In[28]:


if __name__ == "__main__":
    all_datasets_list, tasks_num = AllDatasetsLoader("ECFP", ecfp_dataset_path)
    DatasetsSmiles(all_datasets_list)
    print(get_train_val_fold(tasks_num, fold, k_fold))

