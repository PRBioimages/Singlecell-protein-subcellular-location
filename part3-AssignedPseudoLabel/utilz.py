import os
import pandas as pd
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative"]
INT_2_STR = {x: LBL_NAMES[x] for x in np.arange(19)}
STR_2_INT = {v: k for k, v in INT_2_STR.items()}


def all_Img_df():
    meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
    files = ['hpa_multi_bbox_meta', 'hpa_single_bbox_meta', 'train_single_bbox_meta', 'train_multi_bbox_meta', 'SCV_notest', 'Extra_meta']
    # files = ['hpacell_multi_bbox_meta']
    df_allcell = pd.concat([pd.read_csv(os.path.join(meta_dir, cf + '.csv')) for cf in files], axis=0).reset_index(drop=True)
    df_allcell = df_allcell.drop_duplicates('ID', keep='last', ignore_index=True).reset_index(drop=True)
    return df_allcell


def getalldf():
    meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../")), 'data_csv')
    files = ['hpacell_multi_bbox_meta', 'hpacell_single_bbox_meta', 'traincell_single_bbox_meta', 'traincell_multi_bbox_meta', 'SCVcell_notest_bbox_meta', 'Extracell_bbox_meta']
    # files = ['hpacell_multi_bbox_meta']
    df_allcell = pd.concat([pd.read_csv(os.path.join(meta_dir, cf + '.csv')) for cf in files], axis=0).reset_index(drop=True)
    df_allcell = df_allcell.drop_duplicates('ID', keep='last', ignore_index=True).reset_index(drop=True)
    df_allcell = renamecolumn(df_allcell)

    # df_notest = pd.read_csv('/home/xlzhu/Work1_SingleCellPrediciton/code/work8_singlecell/dataloaders/split/hpa_train_notest.csv')
    df_test = pd.read_csv(join(meta_dir, 'RandomSelect_MamualReset.csv'))
    df_allcell = df_allcell[~df_allcell.ID.isin(df_test.ID)].reset_index(drop=True)
    return df_allcell.reset_index(drop=True)


def renamecolumn(df_allcell):
    if 'idx' not in df_allcell.columns:
        df_allcell.rename(columns={'ID': 'ID_idx'}, inplace=True)
        df_allcell['ID'] = df_allcell['ID_idx'].str.split('_').str[:-1]
        df_allcell['ID'] = df_allcell['ID'].str.join('_')
        df_allcell['idx'] = df_allcell['ID_idx'].str.split('_').str[-1]
    if 'idx' in df_allcell.columns and 'ID_idx' not in df_allcell.columns:
        df_allcell['idx'] = df_allcell['idx'].apply(lambda x: str(x))
        df_allcell['ID_idx'] = df_allcell['ID'].str.cat([df_allcell['idx']], sep='_')
    if 'class0' in df_allcell.columns:
        df_allcell[LBL_NAMES] = df_allcell[[f'class{i}' for i in range(19)]]

    return df_allcell
