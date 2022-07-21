from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


def filter_label(x,i):
    if i in x['Label']:
        return True
    else:
        return False


def main():
    Result_TestDir = './Results'
    df = pd.read_csv(join(Result_TestDir, f'scv_heuristic.csv'))
    df['Label'] = df['Label'].str.split('|').apply(lambda x: [int(i) for i in x])

    for i in range(18):

        sub_df = df.apply(lambda x: filter_label(x, i), axis=1)
        sub_df = df.loc[sub_df.isin([True]), :].reset_index(drop=True)
        sub_df[f'class_selected'] = ''
        IDs = sub_df.ID.unique().tolist()
        k = 0
        select_class = f'class{i}'
        print('Assigning pseudo label for:', select_class)
        for ID in IDs:
            Intensitys = []
            Inds = []
            df_sub = sub_df[sub_df.ID.isin([ID])].reset_index(drop=True)
            for _, sub in df_sub.iterrows():
                Intensitys.append(sub[select_class])
                Inds.append(sub['idx'])
            Intensitys = np.array(Intensitys)
            Intensitys_Inds = list(zip(Inds, Intensitys))
            Intensitys_Inds = np.array(Intensitys_Inds, np.float)
            Intensitys_Inds = sorted(Intensitys_Inds, key=lambda x: x[1])
            # Intensitys_max = np.median(np.array(Intensitys_Inds)[:, 1])
            Intensitys_max = np.mean(np.array(Intensitys_Inds)[-int(len(Intensitys_Inds) / 4):, 1])
            if Intensitys_max == 0:
                Intensitys_max = np.mean(np.array(Intensitys_Inds)[:, 1])
            multi = 0.35
            pseudo_label = np.where(Intensitys > Intensitys_max * multi, 1, 0)
            sub_df.loc[k: k + len(Intensitys_Inds) - 1, f'class_selected'] = pseudo_label
            k = k + len(Intensitys_Inds)
        # sub_df = sub_df[sub_df[select_class] > 0.0].reset_index(drop=True)

        sub_df[select_class] = sub_df[f'class_selected']
        sub_df.to_csv(join(Result_TestDir, f'heuristic_{i}.csv'),
                  index=False)


if __name__ == '__main__':
    main()

