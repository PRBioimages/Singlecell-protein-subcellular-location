import numpy as np
import os
import pandas as pd
from pathlib import Path
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import random
import torch



hpa_mean = [0.0994, 0.0466, 0.0606, 0.0879]
hpa_std = [0.1406, 0.0724, 0.1541, 0.1264]

# stand_mean = [0.485, 0.456, 0.406, 0.406]
# stand_std = [0.229, 0.224, 0.225, 0.225]

LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative"]
INT_2_STR = {x: LBL_NAMES[x] for x in np.arange(19)}
STR_2_INT = {v: k for k, v in INT_2_STR.items()}


def a_ordinary_collect_method(batch):
    '''
    I am a collect method for User Dataset
    '''
    img, pe, exp, msk, cnt = [], [], [], [], []
    # debug
    study_id = []
    weight = []
    # debug end
    if len(batch[0]) == 5:
        for i, p, e, m, l in batch:
            img.append(i)
            pe.append(p)
            exp.append(e)
            msk.append(m)
            cnt.append(l)
        return (torch.cat(img),
                torch.tensor(np.concatenate(exp)).float(), torch.tensor(np.concatenate(msk)).float(), cnt[0])



def all_Img_df():
    meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
    files = ['hpa_multi_bbox_meta', 'hpa_single_bbox_meta', 'train_single_bbox_meta', 'train_multi_bbox_meta', 'SCV_notest']
    # files = ['hpacell_multi_bbox_meta']
    df_allcell = pd.concat([pd.read_csv(os.path.join(meta_dir, cf + '.csv')) for cf in files], axis=0).reset_index(drop=True)
    df_allcell = df_allcell.drop_duplicates('ID', keep='last', ignore_index=True).reset_index(drop=True)
    return df_allcell


class MILDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train', file_dict=None, p=None):
        self.df_img = all_Img_df()
        self.root = cfg.data.data_root
        self.df = df.reset_index(drop=True)
        self.IDs = df.ID.unique()
        self.mode = mode
        self.transform = tfms
        # target_cols = self.df.iloc[:, 1:12].columns.tolist()
        # self.labels = self.df[target_cols].values
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
            # Normalize(mean=hpa_mean, std=hpa_std),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.file_dict = file_dict
        self.cols = LBL_NAMES
        self.pseodu = p
        self.celllabel = cfg.data.celllabel
        print('[ ! ] using', cfg.data.celllabel)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        if self.mode == 'train':
            ID = self.IDs[index]
            row = self.df[self.df.ID.isin([ID])].reset_index(drop=True)
            cnt = self.cfg.experiment.count
            if len(row) > cnt:
                selected = random.sample([i for i in range(len(row))], cnt)
            else:
                selected = [i for i in range(len(row))]
                # selected = random.sample([i for i in range(len(row))], len(row))
                # supp = []
            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19))
            img_label = self.df_img[self.df_img.ID.isin([ID])].reset_index(drop=True)
            img_label = np.max(img_label.loc[0:, self.cols].values.astype(np.float), axis=0)
            for idx, s in enumerate(selected):
                # path = self.path / f'../../{self.cell_path}/{row["ID"]}_{s+1}.png'
                # img = imread(path)
                path = self.root + f'/{row.loc[s, "ID_idx"]}.npy'
                img = np.load(path)
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']
                if not img.shape[0] == self.cfg.transform.size:
                    img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
                # if not len(row) >= cnt:
                #     if len(supp) < cnt - len(row):
                #         img_sup = img.copy()
                #         img_sup[:,:,1] = 0
                #         img_sup = self.tensor_tfms(img_sup)
                #         supp.append(img_sup)
                img = self.tensor_tfms(img)
                batch[idx, :, :, :] = img
                mask[idx] = 1
                if self.celllabel != 'imagelabel':
                    label[idx] = row.loc[s, self.cols].values.astype(np.float)
                else:
                    label[idx] = img_label

            # if not len(row) >= cnt:
            #     supp = torch.stack(supp)
            #     batch[-len(supp):, :, :, :] = supp
            # img_label = self.df_img[self.df_img.ID.isin([ID])].reset_index(drop=True)
            # img_label = np.max(img_label.loc[0:, self.cols].values.astype(np.float), axis=0)
            # img_label = np.max(row.loc[:, self.cols].values.astype(np.float), axis=0)
            # img = self.tensor_tfms(img)
            return batch, label, img_label


            # return batch, mask, label, row[self.cols].values.astype(np.float)

        if self.mode == 'valid':
            ID = self.IDs[index]
            row = self.df[self.df.ID.isin([ID])].reset_index(drop=True)
            selected = [i for i in range(len(row))]
            cnt = len(row)
            batch = torch.zeros((cnt, 4, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19))
            for idx, s in enumerate(selected):
                # path = self.path / f'../../{self.cell_path}/{row["ID"]}_{s+1}.png'
                # img = imread(path)
                path = self.root + f'/{row.loc[s, "ID_idx"]}.npy'
                img = np.load(path)
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']
                if not img.shape[0] == self.cfg.transform.size:
                    img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
                img = self.tensor_tfms(img)
                batch[idx, :, :, :] = img
                mask[idx] = 1
                label[idx] = row.loc[s, self.cols].values.astype(np.float)
            # img = self.tensor_tfms(img)
            # print(cnt)
            # print(row[self.cols].values.astype(np.float))
            # print(cnt)
            img_label = self.df_img[self.df_img.ID.isin([ID])].reset_index(drop=True)
            img_label = np.max(img_label.loc[0:, self.cols].values.astype(np.float), axis=0)
            # img_label = np.max(row.loc[:, self.cols].values.astype(np.float), axis=0)
            return batch, label, img_label, cnt
