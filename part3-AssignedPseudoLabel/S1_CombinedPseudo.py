import pandas as pd
import numpy as np
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *
from utilz import *


def main():
    Clustering_dir = os.path.abspath(os.path.join(os.getcwd(), "./Clustering/Cluster_results/scv/summary/"))
    Clustering_paths = subfiles(Clustering_dir, join=False)
    train = getalldf()

    ## Clustering pseudo labels to update training set labels  ##
    print('updating data by Clustering method')
    for Clustering_path in tqdm(Clustering_paths, total=len(Clustering_paths)):
        l = [int(i) for i in Clustering_path.split('_') if str.isdigit(i)]
        Subcell_loc = list(map(lambda x: str(INT_2_STR[int(x)]), l))
        updatelabel = ['kmeans' + '_' + i for i in Subcell_loc]
        Clustering_df = pd.read_csv(join(Clustering_dir, Clustering_path))
        Clustering_df = Clustering_df.drop_duplicates('ID_idx', keep='last', ignore_index=True).reset_index(drop=True)
        print(f'[ i ] Processing Clustering pseudo csv path, {Clustering_path} to update training set')
        Clustering_df = renamecolumn(Clustering_df)
        temp = Clustering_df.loc[:, ['ID_idx'] + updatelabel]
        temp.columns = ['ID_idx'] + [i for i in Subcell_loc]
        temp = temp.set_index('ID_idx')
        train = train.set_index('ID_idx')
        train.update(temp)
        train.reset_index(inplace=True)


    ## Heuristic pseudo labels to update training set labels  ##
    print('updating data by Heuristic method')
    Heuristic_dir = os.path.abspath(os.path.join(os.getcwd(), "./Heuristic/Results"))
    Heuristic_Label = [11, 15, 12, 1, 9]
    for h_label in Heuristic_Label:
        print(f'Class {INT_2_STR[h_label]} using heuristic method to assign pseudo labels')
        heuristic_data = pd.read_csv(join(Heuristic_dir, f'heuristic_{h_label}.csv'))
        heuristic_data = renamecolumn(heuristic_data)
        class_name = INT_2_STR[h_label]
        temp = heuristic_data.loc[:, ['ID_idx', class_name]]
        temp = temp.set_index('ID_idx')
        train = train.set_index('ID_idx')
        train.update(temp)
        train.reset_index(inplace=True)
        # train[train['ID'].isin(heuristic_data['ID'])]['fold'] = -1

    train = train[['ID_idx', 'ID', 'idx', 'fold'] + LBL_NAMES].reset_index(drop=True)
    meta_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'data_csv')
    train.to_csv(join(meta_dir, f'all_cell_pseudo.csv'), index=False)


if __name__ == '__main__':
    main()
