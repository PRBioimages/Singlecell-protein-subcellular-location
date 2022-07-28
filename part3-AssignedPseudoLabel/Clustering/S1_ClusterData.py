import numpy as np
import os
import cv2
import pandas as pd
import mlcrate as mlc
from scipy.stats.mstats import mquantiles
from util import *


def mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print('---the '+dirName+' is created!---')
    else:
        print('---The dir is there!---')


def transform(par):
    subdf, img_dir = par
    ID = subdf['ID']
    idx = subdf['idx']
    npy = ID + F'_{idx}.npy'
    npy_path = os.path.join(npy_dir, npy)
    # print(npy_path)
    data = np.load(npy_path)
    img_path = os.path.join(img_dir, npy.replace('.npy', ''))
    for i in range(4):
        if mquantiles(data[..., i].flatten(), 0.999) == 0:
            return
    cv2.imwrite(img_path + '_red.png', data[..., 0])
    cv2.imwrite(img_path + '_green.png', data[..., 1])
    cv2.imwrite(img_path + '_blue.png', data[..., 2])
    cv2.imwrite(img_path + '_yellow.png', data[..., 3])


if __name__ == '__main__':
    npy_dir = '../HPA_data/Cell/data'
    pool = mlc.SuperPool(8)

    for n in range(0, 18):
        # n = 1
        img_dir = f'./IFCellDataset/normal/{n}'
        mkdir(img_dir)
        df = getalldf()
        # single cell
        meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
        dfcsv = pd.read_csv(os.path.join(meta_dir, 'SCV_notest.csv'))
        # df = df.iloc[(n-1) * 10000: n * 10000, :]
        # single cell
        df.Label = df.Label.apply(lambda x: str(x))
        df = df[~df.ID.isin(dfcsv.ID)]

        INT_loc = f'{n}'
        print('INT_loc', INT_2_STR[n])
        df = df[df.Label.isin([INT_loc])].reset_index(drop=True)
        n = np.minimum(len(df), 1000)
        df = df.sample(n=n, replace=False, random_state=12345).reset_index(drop=True)
        par = [(subdf, img_dir) for _, subdf in df.iterrows()]
        # npy_list = os.listdir(npy_dir)

        pool.map(transform, par, description='get sing cell images(.png)')

    # scv
    img_dir = f'./IFCellDataset/scv/'
    mkdir(img_dir)
    meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
    df = pd.read_csv(os.path.join(meta_dir, 'SCVcell_notest_bbox_meta.csv'))
    if not 'idx' in df.columns:
        df.rename(columns={'ID': 'ID_idx'}, inplace=True)
        df['ID'] = df['ID_idx'].str.split('_').str[:-1]
        df['ID'] = df['ID'].str.join('_')
        df['idx'] = df['ID_idx'].str.split('_').str[-1]

    print('INT_loc: SCV info')
    par = [(subdf, img_dir) for _, subdf in df.iterrows()]
    # npy_list = os.listdir(npy_dir)
    pool.map(transform, par, description='get sing cell images(.png)')
