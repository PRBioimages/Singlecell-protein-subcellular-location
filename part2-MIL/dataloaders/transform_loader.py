import albumentations as A
import os


def get_tfms(name):
    path = os.path.dirname(os.path.realpath(__file__)) + f'/../configs/{name}.yaml'
    return A.load(path, data_format='yaml')


if __name__ == '__main__':
    get_tfms('augmentation')
