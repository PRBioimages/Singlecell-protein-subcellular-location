import argparse
from configs import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-config', default='config.yaml', type=str, help='path of config file(.yaml)')
    arg('--seed', type=int, default=-1)
    args = parser.parse_args()
    cfg = get_config(args.config)


    return cfg

