import argparse
import os
import torch
import numpy as np
import random
from configs import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-config', default='config.yaml', type=str, help='path of config file(.yaml)')
    args = parser.parse_args()

    # load the config file
    cfg = get_config(args.config)

    # Initial jobs
    # set seed, should always be done
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)

    # set the gpu to use
    print('[ √ ] Using #{} GPU'.format(cfg.basic.GPU))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join(cfg.basic.GPU))
    # print(','.join(args.gpu))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return cfg

