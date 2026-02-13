import os, random, time
import torch
import numpy as np
import pandas as pd
import logging
from logging import Logger


def mksure_path(dirs_or_files):
    if not os.path.exists(dirs_or_files):
        os.makedirs(dirs_or_files)



def set_seed(seed=0):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    # print("set seed %d" % seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_checkpoint_dir(save_path):
    current_time = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    checkpoint_path = save_path + 'checkpoints_' + current_time
    mksure_path(checkpoint_path)

    return checkpoint_path
