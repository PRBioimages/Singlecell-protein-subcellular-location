from batchgenerators.utilities.file_and_folder_operations import *
from skimage import filters,segmentation,measure,morphology,color
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell
from tqdm import tqdm
import cv2
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#### You need to download the HPA segmentation model in advance, and fill in the path of the model below. ###

NUC_MODEL = './nuclei-model.pth'
CELL_MODEL = './cell-model.pth'


segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device="cuda:0",
    padding=True,
    multi_channel_model=True)


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


def GettingInfoWidthAndNum(itterows, root_data):
    ID = itterows['ID']
    segment_path = os.path.join(root_data, ID + '_red.png')
    mask = cv2.imread(segment_path, 0)
    ImageWidth = mask.shape[-1]
    return ImageWidth


def SegmentingKaggle():
    '''Segmenting IF iamges to get cell masks'''

    image_path = '../HPA_data/IF/data'
    root_mask = '../HPA_data/IF/mask'


    kaggle_df_path = './csv/train.csv'
    Kaggle_df = pd.read_csv(kaggle_df_path)
    Split_csv2_Single_and_multi(Kaggle_df, 'train')

    if 'ImageWidth' not in Kaggle_df.columns:
        Kaggle_df['ImageWidth'] = 0
        tqdm.pandas(desc="Get image width")
        Kaggle_df['ImageWidth'] = Kaggle_df.progress_apply(lambda x: GettingInfoWidthAndNum(x, image_path), axis=1)
        Kaggle_df.to_csv('./csv/train.csv', index=False)

    colors = ['blue', 'red', 'yellow']


    BATCH_SIZE = 8
    IMAGE_SIZES = [1728, 2048, 3072, 4096]
    predict_df_1728 = Kaggle_df[Kaggle_df.ImageWidth == IMAGE_SIZES[0]]
    predict_df_2048 = Kaggle_df[Kaggle_df.ImageWidth == IMAGE_SIZES[1]]
    predict_df_3072 = Kaggle_df[Kaggle_df.ImageWidth == IMAGE_SIZES[2]]
    predict_df_4096 = Kaggle_df[Kaggle_df.ImageWidth == IMAGE_SIZES[3]]

    predict_ids_1728 = predict_df_1728.ID.to_list()
    predict_ids_2048 = predict_df_2048.ID.to_list()
    predict_ids_3072 = predict_df_3072.ID.to_list()
    predict_ids_4096 = predict_df_4096.ID.to_list()

    for size_idx, submission_ids in enumerate([predict_ids_1728, predict_ids_2048, predict_ids_3072, predict_ids_4096]):
        size = IMAGE_SIZES[size_idx]
        if submission_ids == []:
            print(f"\n...SKIPPING SIZE {size} AS THERE ARE NO IMAGE IDS ...\n")
            continue
        else:
            print(f"\n...WORKING ON IMAGE IDS FOR SIZE {size} ...\n")
        for i in tqdm(range(0, len(submission_ids), BATCH_SIZE), total=int(np.ceil(len(submission_ids) / BATCH_SIZE))):

            Id = [id for id in submission_ids[i:(i + BATCH_SIZE)]]
            Id_used = [id for id in Id if not os.path.exists(join(root_mask, id+'_mask.png'))]
            if Id_used == []:
                continue
            label_paths = []
            red_imgs, yellow_imgs, blue_imgs = [], [], []
            for id in Id_used:
                label_path = join(root_mask, id+'_mask.png')
                exec(f'label_paths.append(label_path)')
                for color in colors:
                    # img = img.replace('images.', '')
                    img_name = f'{id}_{color}.png'
                    exec(f'{color}_path = join(image_path, img_name)')
                    a = eval(f'{color}_path')
                    img = cv2.imread(a, 0)
                    # exec(f'{color}_img = np.array(Image.open({color}_path).convert(%s))'% mode)
                    exec(f'{color}_imgs.append(img)')


            cell_segmentations = segmentator.pred_cells([red_imgs, yellow_imgs, blue_imgs])
            nuc_segmentations = segmentator.pred_nuclei(blue_imgs)
            cell_mask = [label_cell(nuc_seg, cell_seg)[1].astype(np.uint8) for nuc_seg, cell_seg in
                         zip(nuc_segmentations, cell_segmentations)]
            for l in range(len(label_paths)):
                cv2.imwrite(label_paths[l], cell_mask[l])


def SegmentingHPA():
    image_path = '../HPA_data/IF/cache_gz/data'
    root_mask = '../HPA_data/IF/mask'

### Some data in the HPA database have no mask, and this part of the image is stored in image_path
    print('\n[!] The image to be segmented is stored in : ', image_path)
    Imagefiles = listdir(image_path)
    Imagefiles = pd.Series(Imagefiles)
    Imagefiles = Imagefiles.str.split('_').str[:-1]
    Imagefiles = Imagefiles.str.join('_')
    IDs = pd.unique(Imagefiles).tolist()

    colors = ['blue', 'red', 'yellow']
    Id_used = [id for id in IDs if not os.path.exists(join(root_mask, id + '_mask.png'))]
    for i in tqdm(Id_used, total=len(Id_used)):
        if Id_used == []:
            break
        label_paths = []
        red_imgs, yellow_imgs, blue_imgs = [], [], []
        for id in Id_used:
            label_path = join(root_mask, id + '_mask.png')
            exec(f'label_paths.append(label_path)')
            for color in colors:
                # img = img.replace('images.', '')
                img_name = f'{id}_{color}.png'
                exec(f'{color}_path = join(image_path, img_name)')
                a = eval(f'{color}_path')
                img = cv2.imread(a, 0)
                # exec(f'{color}_img = np.array(Image.open({color}_path).convert(%s))'% mode)
                exec(f'{color}_imgs.append(img)')

        cell_segmentations = segmentator.pred_cells([red_imgs, yellow_imgs, blue_imgs])
        nuc_segmentations = segmentator.pred_nuclei(blue_imgs)
        cell_mask = [label_cell(nuc_seg, cell_seg)[1].astype(np.uint8) for nuc_seg, cell_seg in
                     zip(nuc_segmentations, cell_segmentations)]
        for l in range(len(label_paths)):
            cv2.imwrite(label_paths[l], cell_mask[l])



if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    print('\n[!] Segmenting Kaggle IF images to masks with single cells... ')
    SegmentingKaggle()

    print('\n[!] Segmenting IF images without masks in HPA database... ')
    SegmentingHPA()

    print('\nsuccess!')
