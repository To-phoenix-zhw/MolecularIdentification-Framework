import logging
import requests
deepchem_logger = logging.getLogger('deepchem')
deepchem_logger.setLevel(logging.CRITICAL)
dc_logger1= logging.getLogger('dc')
dc_logger1.setLevel(logging.CRITICAL)
import pickle, time,os
import numpy as np
import math
from optparse import OptionParser
from utils import *
from data import *

from train import *
from test import *




parser = OptionParser()
parser.add_option("--train_path", dest="train_path", default='data/dataset/training-set')
parser.add_option("--val_path", dest="val_path", default='')
parser.add_option("--test_path", dest="test_path", default='')
parser.add_option("--device", dest="device", default='cuda:0')  # you can change it to 'cuda:0'
parser.add_option("--seed", dest="seed", default=0)
parser.add_option("--lstmdim", dest="lstmdim", default=256)
parser.add_option("--search_space", dest="search_space", default=100)
parser.add_option("--searchtimes", dest="searchtimes", default=1)
parser.add_option("--num_iter", dest="num_iter", default=10)
parser.add_option("--optnum", dest="optnum", default=5)
parser.add_option("--maxepoch", dest="maxepoch", default=160000)
parser.add_option("--gamma", dest="gamma", default=0.5)
parser.add_option("--pri", dest="pri", default='false')
parser.add_option("--active_flag", dest="active_flag", default='true')
parser.add_option("--epsilon", dest="epsilon", default=0.1)
parser.add_option("--mode", dest="mode", default='train')
parser.add_option("--custom", dest="custom", default='')
parser.add_option("--case_task", dest="case_task", default=0)
parser.add_option("--checkpoint_path", dest="checkpoint_path", default='')
parser.add_option("--model_path", dest="model_path", default='')
parser.add_option("--continue_epoch", dest="continue_epoch", default=0)
parser.add_option("--continue_rewards", dest="continue_rewards", default=0)
parser.add_option("--continue_distances", dest="continue_distances", default=0)
parser.add_option("--test_times", dest="test_times", default=100)
parser.add_option("--task_id", dest="task_id", default=84)
opts,args = parser.parse_args()

device = str(opts.device)
seed = int(opts.seed) 
lstmdim = int(opts.lstmdim)  
search_space = int(opts.search_space)  
searchtimes = int(opts.searchtimes)  
num_iter = int(opts.num_iter)  
optnum = int(opts.optnum) 
maxepoch = int(opts.maxepoch) 
gamma = float(opts.gamma) 
pri = True if (opts.pri == 'true') else False 
active_flag = True if (opts.active_flag == 'true') else False
epsilon = float(opts.epsilon)  
mode = str(opts.mode)
custom = str(opts.custom)  
case_task = int(opts.case_task)  
model_path = str(opts.model_path) 
continue_epoch = int(opts.continue_epoch)
continue_rewards = float(opts.continue_rewards)
continue_distances = float(opts.continue_distances)
test_path = str(opts.test_path)  
test_times = int(opts.test_times) 
task_id = int(opts.task_id) 
checkpoint_path = opts.checkpoint_path


def load_datasets(directory):
    datasets = []
    names = []

    if not os.path.isdir(directory):
        print(f"{directory} does not exist.")
        return
    data = []
    contents = os.listdir(directory)
    for i,item in enumerate(contents):
        path = directory+'/'+item
        dataset = dc.data.DiskDataset(path)
        ds = dc.data.NumpyDataset.from_DiskDataset(dataset)
        datasets.append(ds)
        names.append(item)
    return datasets, names


if __name__ == "__main__":
    time_start = time.time()
    set_seed(seed)

    if mode == "train":
        # Logging
        logger = get_logger('train', checkpoint_path)
        #logger.info(opts)
        train_datasets_list,datasetnames  = load_datasets(opts.train_path)
        val_datasets_list,_  = load_datasets(opts.val_path)

        rewards_list = train(
            train_datasets_list,
            val_datasets_list,
            checkpoint_path,
            model_path,
            continue_epoch,
            continue_rewards,
            continue_distances,
            mode,
            logger,
            device,
            lstmdim,
            search_space,
            searchtimes, 
            num_iter,
            optnum,
            maxepoch,
            gamma,
            pri,
            active_flag,
            epsilon
        )
        

        record_path = checkpoint_path + "run_result/"
        mksure_path(record_path)
        

        xline=list(np.arange(len(rewards_list)))
        plt.plot(xline,rewards_list)
        plt.savefig(record_path + 'runresult.svg')

        with open(record_path + "rewards_list.txt", "wb") as fp:  
            pickle.dump(rewards_list, fp)

        time_end = time.time()
        print('time cost', time_end - time_start, 's')


    elif mode == "test":
        dataset_num = task_id
        datasets_list,datasetnames  = load_datasets(test_path)
        datasets_list= datasets_list[dataset_num]

        try:
            al_find = 0
            al_re = 0
            al_avgsota = 0
            al_dis = 0
            al_step = 0
            print("Task: ", datasetnames[dataset_num])

            for i in range(test_times):  
                print("*"*8 + "Test " + str(i) + "*"*8)
                alre, alsota, aldis, alstep, identify_flag = test(
                    datasets_list,
                    dataset_num,
                    opts.checkpoint_path,
                    mode,
                    device,
                    lstmdim,
                    search_space,
                    searchtimes, 
                    num_iter,
                    optnum,
                    maxepoch,
                    gamma,
                    pri,
                    active_flag,
                    epsilon
                )
                if identify_flag:
                    al_find += 1

                al_re += alre
                al_avgsota += alsota
                al_dis += aldis
                al_step += alstep

            print("*"*8 + "Statistic Performance" + "*"*8)
            print("Task: ", datasetnames[dataset_num])
            print("Average success rate: %.2f%%" % ((al_find/test_times)*100))
            print("Average search steps: %d" % (math.ceil(al_step/test_times)))

        except KeyboardInterrupt:
            print("Terminating...")

        time_end = time.time()
        print('time cost', time_end - time_start, 's')

    elif mode == "case":
        dataset_num = task_id
        datasets_list,datasetnames  = load_datasets(test_path)
        datasets_list= datasets_list[dataset_num]

        try:
            al_find = 0
            al_re = 0
            al_avgsota = 0
            al_dis = 0
            al_step = 0
            print("Task: ", datasetnames[dataset_num])

            alre, alsota, aldis, alstep, identify_flag = test(
                datasets_list,
                dataset_num,
                opts.checkpoint_path,
                mode,
                device,
                lstmdim,
                search_space,
                searchtimes, 
                num_iter,
                optnum,
                maxepoch,
                gamma,
                pri,
                active_flag,
                epsilon
            )
        except KeyboardInterrupt:
            print("Terminating...")

        time_end = time.time()
        print('time cost', time_end - time_start, 's')





    else:
        raise Exception("mode error!!!") 

