import os
import pandas as pd
import numpy as np


LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative"]
INT_2_STR = {x: LBL_NAMES[x] for x in np.arange(19)}
STR_2_INT = {v: k for k, v in INT_2_STR.items()}


def getalldf():
    meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
    files = ['hpacell_multi_bbox_meta', 'hpacell_single_bbox_meta', 'traincell_single_bbox_meta', 'traincell_multi_bbox_meta', 'SCVcell_notest_bbox_meta']
    # files = ['hpacell_multi_bbox_meta']
    df_allcell = pd.concat([pd.read_csv(os.path.join(meta_dir, cf + '.csv')) for cf in files], axis=0).reset_index(drop=True)
    df_allcell.rename(columns={'ID': 'ID_idx'}, inplace=True)
    df_allcell['ID'] = df_allcell['ID_idx'].str.split('_').str[:-1]
    df_allcell['ID'] = df_allcell['ID'].str.join('_')
    df_allcell['idx'] = df_allcell['ID_idx'].str.split('_').str[-1]
    df_allcell = df_allcell.drop_duplicates('ID_idx')
    return df_allcell.reset_index(drop=True)


def preprocessing_df(df_path):
    SLF_Feature_cells = pd.read_csv(df_path,)
    features_name = list(range(SLF_Feature_cells.shape[-1]-1))
    c = SLF_Feature_cells.columns
    if 'Label' not in c:
        SLF_Feature_cells = pd.read_csv(df_path, header=None, names=['ID_idx'] + features_name)
        df_allcell = getalldf()
        SLF_Feature_cells = pd.merge(SLF_Feature_cells, df_allcell, on='ID_idx')
        SLF_Feature_cells.replace([np.inf, -np.inf], np.nan, inplace=True)
        dropcolum = [i for i in features_name if any(SLF_Feature_cells[i] > 100000)]
        SLF_Feature_cells.drop(columns=dropcolum, inplace=True)
        SLF_Feature_cells.dropna(how='any', axis=1, inplace=True)
        features_name = list(set(features_name).intersection(SLF_Feature_cells.columns))
        dropcolum = [i for i in features_name if SLF_Feature_cells[i].sum(axis=0) == 0]
        SLF_Feature_cells.drop(columns=dropcolum, inplace=True)
        features_name = list(set(features_name).intersection(SLF_Feature_cells.columns))
        SLF_Feature_cells = SLF_Feature_cells.loc[:, ['ID_idx', 'Label']+features_name]
        # SLF_Feature_cells.columns = ['ID_idx', 'Label'] + list(range(SLF_Feature_cells.shape[-1]-2))
        # SLF_Feature_cells.to_csv(df_path, index=False, na_rep='NaN')
    return SLF_Feature_cells


def MatchLabel(cluster, target):
    new_cluster = []
    cluster_label = np.unique(cluster)
    idx_allcluster = [np.where(cluster == i)[0] for i in cluster_label]
    for _, idx in enumerate(idx_allcluster):
        cluster_match_target = target[idx].value_counts().sort_values().index[-1]
        new_cluster.append(cluster_match_target)
    if len(np.unique(new_cluster)) < len(cluster_label):
        new_cluster = []
        for _, idx in enumerate(idx_allcluster):
            cluster_match_target = target[idx].value_counts().sort_index()
            new_cluster.append(cluster_match_target / cluster_match_target.sum())
        new_cluster = np.argmax(np.array(new_cluster), 0)
        new_cluster = [i for i in new_cluster]
    return new_cluster