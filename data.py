import random
import numpy as np
import pandas as pd
import deepchem as dc


class Point(object):
    def __init__(self, idx, X, y, smi):
        self.idx = idx
        self.X = X
        self.y = y
        self.smi = smi
    def __str__(self):
        return "{}, {}".format(self.y, self.smi)




def load_train_molnet_datasets():
    train_molnet_datasets = []
    molnet_root = "./data/datasets/molnet/normECFP/"
    train_tasks_df = pd.read_excel(molnet_root + 'train_tasks.xlsx')
    train_tasks = train_tasks_df.values.squeeze().tolist()
    
    for part_concat in train_tasks:
        train_molnet_datasets.append(dc.data.DiskDataset(molnet_root + part_concat))
    return train_molnet_datasets


def load_test_molnet_datasets():
    test_molnet_datasets = []
    molnet_root = "./data/datasets/molnet/normECFP/"
    test_tasks_df = pd.read_excel(molnet_root + 'test_tasks.xlsx')
    test_tasks = test_tasks_df.values.squeeze().tolist()
    
    for part_concat in test_tasks:
        test_molnet_datasets.append(dc.data.DiskDataset(molnet_root + part_concat))
    return test_molnet_datasets


def load_train_chembl_datasets():
    train_chembl_datasets = []
    chembl_root = "./data/datasets/ChEMBL/"
    pre_part = "./data/datasets/ChEMBL/norm_pValue/"
    train_address = pd.read_csv(chembl_root + 'train_address.csv').squeeze().values.tolist()
    
    for part_concat in train_address:
        train_chembl_datasets.append(dc.data.DiskDataset(pre_part + part_concat))
    return train_chembl_datasets


def load_test_chembl_datasets():
    test_chembl_datasets = []
    chembl_root = "./data/datasets/ChEMBL/"
    pre_part = "./data/datasets/ChEMBL/norm_pValue/"
    test_address = pd.read_csv(chembl_root + 'test_address.csv').squeeze().values.tolist()
    
    for part_concat in test_address:
        test_chembl_datasets.append(dc.data.DiskDataset(pre_part + part_concat))
    return test_chembl_datasets


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
        
    # print("Quantum_Mechanical_Datasets: ")
    
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
    
    # print("Physical_Chemistry_Datasets: ")
    
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
    
    # print("BioDatasets: ")
    
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
    
    # print("Physiology_Datasets")
    
    return physi_datasets_list


# In[11]:


def AllDatasetsLoader(featurizer, dataset_path):
    # Four categories
    quantum_datasets_list = Quantum_Mechanical_Datasets(featurizer, dataset_path)
    pc_datasets_list = Physical_Chemistry_Datasets(featurizer, dataset_path)
    bio_datasets_list = BioDatasets(featurizer, dataset_path)
    physi_datasets_list = Physiology_Datasets(featurizer, dataset_path)
    
    # Merge them
    all_datasets_list = quantum_datasets_list + pc_datasets_list + bio_datasets_list + physi_datasets_list
    
    return all_datasets_list



def load_full_train_datasets():
    train_molnet_datasets = load_train_molnet_datasets()
    train_chembl_datasets = load_train_chembl_datasets()
    train_datasets_list = train_molnet_datasets + train_chembl_datasets
    return train_datasets_list


def load_full_test_datasets():
    test_molnet_datasets = load_test_molnet_datasets()
    test_chembl_datasets = load_test_chembl_datasets()
    test_datasets_list = test_molnet_datasets + test_chembl_datasets
    return test_datasets_list


def load_train_datasets():
    train_chembl_datasets = load_train_chembl_datasets()
    return train_chembl_datasets


def load_test_datasets():
    ecfp_dataset_path = "./data/datasets/molnet/ECFP/"
    featurizer="ECFP"
    datasets_list = AllDatasetsLoader(featurizer, ecfp_dataset_path)
    val_nos = [10, 32, 42]
    datasets_name = ['QM8: quantum mechanical calculations', 'Thermosol: the thermodynamic solubility dataset', 'BACE: quantitative IC50 binding results for inhibitors of human beta-secretase 1']
    return datasets_list, datasets_name, val_nos



def get_obj_task(task_obj, datasets_list):
    task_idx = 0  
    y_obj_no = -1
    dataset_obj_idx = -1
    for dataset_idx, dataset_element in enumerate(datasets_list):
        tasks = dataset_element.get_task_names()
        for task_no, task in enumerate(tasks):
            if task_idx == task_obj:
                dataset_obj_idx = dataset_idx
                y_obj_no = task_no
                break
            task_idx += 1
        if y_obj_no != -1:
            break

    return dataset_obj_idx, y_obj_no


def compute_extreme_and_initial_point(X, y, ids, search_space=100):
    """Calculate the points corresponding to the maximum and minimum true values. 
    Random initialize a starting point."""
    
    X_dim = X.shape[1] 
    maxy_index = np.argmax(y)
    maxy_X = X[maxy_index].reshape(-1, X_dim)  
    maxy_y = y[maxy_index].reshape(-1, 1) 
    maxy_id = ids[maxy_index]  
    GT_max_point = Point(maxy_index, maxy_X, maxy_y, maxy_id)
    # print("GT max Point: ", GT_max_point.smi)
    # print(GT_max_point.y.item())

    miny_index = np.argmin(y)
    miny_X = X[miny_index].reshape(-1, X_dim)
    miny_y = y[miny_index].reshape(-1, 1)
    miny_id = ids[miny_index]
    GT_min_point = Point(miny_index, miny_X, miny_y, miny_id)
    # print("GT min Point: ", GT_min_point.smi)
    # print(GT_min_point.y.item())


    randomindex = random.randint(0, search_space-1)
    initial_X = X[randomindex].reshape(-1, X_dim)  # [1, 1024]
    initial_y = y[randomindex].reshape(-1, 1)  # [1, 1]
    initial_id = ids[randomindex]  # a string
    initial_point = Point(randomindex, initial_X, initial_y, initial_id)
    # print("Random Chosen Initial Point", initial_point.smi)
    # print(initial_point.y.item())
    
    return GT_max_point, GT_min_point, initial_point 



def compute_extreme_and_initial_point_appoint(X, y, ids, initial=0):
    """Calculate the points corresponding to the maximum and minimum true values. 
    Specify a molecule as the initial point."""
    X_dim = X.shape[1]  
    maxy_index = np.argmax(y)
    maxy_X = X[maxy_index].reshape(-1, X_dim)  # [1, 1024]
    maxy_y = y[maxy_index].reshape(-1, 1)  # [1, 1]
    maxy_id = ids[maxy_index]  # a string
    GT_max_point = Point(maxy_index, maxy_X, maxy_y, maxy_id)
    # print("GT max Point: ", GT_max_point.smi)
    # print(GT_max_point.y.item())

    miny_index = np.argmin(y)
    miny_X = X[miny_index].reshape(-1, X_dim)
    miny_y = y[miny_index].reshape(-1, 1)
    miny_id = ids[miny_index]
    GT_min_point = Point(miny_index, miny_X, miny_y, miny_id)
    # print("GT min Point: ", GT_min_point.smi)
    # print(GT_min_point.y.item())

    randomindex = initial
    initial_X = X[randomindex].reshape(-1, X_dim)  # [1, 1024]
    initial_y = y[randomindex].reshape(-1, 1)  # [1, 1]
    initial_id = ids[randomindex]  # a string
    initial_point = Point(randomindex, initial_X, initial_y, initial_id)
    # print("Chosen Initial Point", initial_point.smi)
    # print(initial_point.y.item())
    
    return GT_max_point, GT_min_point, initial_point 
