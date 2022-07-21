import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--IFdata_root', type=str, help='the root directory of the cell dataset')
    arg('--Maskdata_root', type=str, help='the root directory of the cell dataset')
    arg('-config', type=str, help='path of config file(.yaml)')
    arg('-model_path', type=str, help='path of MIL model(.pth)')
    arg('--seed', type=int, default=-1)
    args = parser.parse_args()

    return args

