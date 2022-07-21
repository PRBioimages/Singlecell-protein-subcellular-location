import argparse
import os
import torch
import numpy as np
import random
import json
from configs import get_config, Config
from path import Path
import pandas as pd


def load_matched_state(model, state_dict):
    model_dict = model.state_dict()
    not_loaded = []
    for k, v in state_dict.items():
        if k in model_dict.keys():
            if not v.shape == model_dict[k].shape:
                print('Error Shape: {}, skip!'.format(k))
                continue
            model_dict.update({k: v})
        else:
            # print('not matched: {}'.format(k))
            not_loaded.append(k)
    if len(not_loaded) == 0:
        print('[ √ ] All layers are loaded')
    else:
        print('[ ! ] {} layer are not loaded'.format(len(not_loaded)))
    model.load_state_dict(model_dict)


def prepare_for_result(cfg: Config):
    print('model storage directory: ', cfg.train.dir)
    if not os.path.exists(cfg.train.dir):
        raise Exception('Result dir not found')
    if os.path.exists(cfg.train.dir + '/' + cfg.model.name + '-' + cfg.data.celllabel):
        if cfg.basic.debug:
            print('[ X ] The output dir already exist!')
            output_path = Path(cfg.train.dir) / cfg.basic.id
            return output_path
        else:
            raise Exception('The output dir already exist')
    output_path = Path(cfg.train.dir) / cfg.model.name + '-' + cfg.data.celllabel
    os.mkdir(output_path)
    os.mkdir(output_path / 'checkpoints')
    with open(output_path / 'train.log', 'w') as fp:
        fp.write(
            'Epochs\tlr\ttrain_loss\n'
        )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--debug', type=bool, default=False, help='if true, enter the debug mode')
    arg('--gpu', type=str, default='0', help='Specify GPU, multi GPU need to be seperated by commas, --gpu 0,1')
    arg('-config', type=str, default='config.yaml', help='path of config file(.yaml)')
    arg('--seed', type=int, default=-1)
    args = parser.parse_args()

    # load the config file
    cfg = get_config(args.config)

    cfg.basic.GPU = [str(i) for i in str(args.gpu).split(',')]
    cfg.basic.debug = args.debug


    if not args.seed == -1:
        cfg.basic.seed = args.seed

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
    return args, cfg



