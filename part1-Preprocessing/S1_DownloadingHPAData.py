import io
import os
import cv2
import requests
import pathlib
import gzip
import imageio
import pandas as pd
import mlcrate as mlc
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import numpy as np



s = requests.session()
s.keep_alive = False


def tif_gzip_to_png(tif_path):
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. for working in local work station
    '''
    png_path = pathlib.Path(tif_path.replace('.tif.gz', '.png'))
    tf = gzip.open(tif_path).read()
    img = imageio.imread(tf, 'tiff')
    imageio.imwrite(png_path, img)


def download_and_convert_tifgzip_to_png(url, target_path):
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. in Kaggle notebook
    '''
    r = requests.get(url)
    f = io.BytesIO(r.content)
    tf = gzip.open(f).read()
    img = imageio.imread(tf, 'tiff')
    imageio.imwrite(target_path, img)


def download_and_convert_jpg_to_png(params):
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. in Kaggle notebook
    '''
    url, target_path = params
    try:
        img = Image.open(target_path)
        img = np.asarray(img.convert('L'))
        s = img.dtype
    except:
        i = 0
        while i < 5:
            try:
                print(url)
                r = requests.get(url, allow_redirects=True)
                open(target_path, 'wb').write(r.content)
                return
            except requests.exceptions.RequestException:
                i += 1


def Split_csv2_Single_and_multi(df, datasource):
    """
    This function divides the .csv file into two .csv files according to the number of labels
    :param df: The .csv file where the function needs to be executed
    :param datasource: The source of the data is kaggle or HPA
    :return: Separated .csv files according to single-label and multi-label
    """
    single_path = f'./csv/{datasource}_single_bbox_meta.csv'
    multi_path = f'./csv/{datasource}_multi_bbox_meta.csv'
    if datasource == 'hpa':
        df['Image'] = df['Image'].str.split("/").str[-1]
        df.rename(columns={'Image': 'ID'}, inplace=True)
    if not os.path.exists(single_path):
        list_a = []
        for i, row in df.iterrows():
            if len(np.unique(row.Label.strip().split('|'))) == 1:
                list_a.append(True)
            else:
                list_a.append(False)
        list_a = pd.Series(list_a)
        df_single = df[list_a.values].reset_index(drop=True)
        df_single.to_csv(single_path, index=False, na_rep='NaN')

    if not os.path.exists(multi_path):
        list_a = []
        for i, row in df.iterrows():
            if len(np.unique(row.Label.strip().split('|'))) > 1:
                list_a.append(True)
            else:
                list_a.append(False)
        list_a = pd.Series(list_a)
        df_multi = df[list_a.values].reset_index(drop=True)
        df_multi.to_csv(multi_path, index=False, na_rep='NaN')


def main():
    public_hpa_df = pd.read_csv('./csv/kaggle_2021.tsv')
    save_dir = os.path.join('../HPA_data/IF/HPAdata')
    maybe_mkdir_p(save_dir)


    colors = ['blue', 'red', 'green', 'yellow']


    # Remove all images overlapping with Training set
    public_hpa_df = public_hpa_df[public_hpa_df.in_trainset == False]

    # Remove all images with only labels that are not in this competition
    public_hpa_df = public_hpa_df[~public_hpa_df.Label_idx.isna()]

    # Remove all images with Cellline that are not in this competition
    celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30',
                 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
    public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines)].reset_index(drop=True)

    public_hpa_df_17.columns = ['Image', 'Label_idx', 'Cellline', 'in_trainset', 'Label']
    Split_csv2_Single_and_multi(public_hpa_df_17, 'hpa')

    params = []
    for img in public_hpa_df_17['Image'].values:
        for color in colors:
            img_url = f'{img}_{color}.jpg'
            save_path = os.path.join(save_dir, f'{os.path.basename(img)}_{color}.png')
            params.append((img_url, save_path))

    pool = mlc.SuperPool(8)
    pool.map(download_and_convert_jpg_to_png, params, description='download hpa v20')


if __name__ == '__main__':
    main()
