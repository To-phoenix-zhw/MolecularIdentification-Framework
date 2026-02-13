import logging
import requests
deepchem_logger = logging.getLogger('deepchem')
deepchem_logger.setLevel(logging.CRITICAL)
dc_logger1= logging.getLogger('dc')
dc_logger1.setLevel(logging.CRITICAL)
import deepchem as dc
import numpy as np
import pandas as pd



from optparse import OptionParser
parser = OptionParser()
parser.add_option("--src_path", dest=None, default=None)
parser.add_option("--dest_path", dest=None, default=None)
parser.add_option("--normlize", dest=None, default=None)
parser.add_option("--column", dest=None, default="Values")
opts,args = parser.parse_args()

def savepath(src_path, dest_path, normlize, column):
    LD_51 = pd.read_excel(src_path)
    pro = LD_51[column].values
    
    if isinstance(pro[0], (str)) : 
        if pro[0]=="minus":
            pro[1:] = np.max(pro[1:])-pro[1:]
        elif pro[0]=="minus-paper":
            pro[1:] = np.max(pro[1:])-pro[1:]
            maxx = np.max(pro[1:])
            minn = np.min(pro[1:])
            pro[1:] = (pro[1:]-minn)/(maxx-minn) 

        elif pro[0]=="norm":
            maxx = np.max(pro[1:])
            minn = np.min(pro[1:])
            pro[1:] = (pro[1:]-minn)/(maxx-minn) 



    data_path = dest_path
    smiles = LD_51["Smiles"].values
    prop,smilesp = [],[]
    for s,p in zip(smiles, pro):
        if isinstance(p, (int, float, complex)) and isinstance(s, (str)) : 
            smilesp.append(s)
            prop.append(p)

    featurizer = dc.feat.CircularFingerprint(size=1024)  # 'ECFP'
    features = featurizer.featurize(smilesp)
    #created_dataset = dc.data.NumpyDataset(X=features, y=pro, ids=smiles)
    created_disk_dataset = dc.data.DiskDataset.from_numpy(X=features, y=prop, ids=smilesp, tasks=["Values"], data_dir=data_path)

savepath(opts.src_path, opts.dest_path, opts.normlize, opts.column)
