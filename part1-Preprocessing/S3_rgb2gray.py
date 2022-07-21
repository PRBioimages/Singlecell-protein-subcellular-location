import cv2
import pandas as pd
from PIL import Image
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *



color_channel = {'red': 0, 'green': 1, 'blue': 2, 'yellow': 0}


def rgb_to_gray(tif_name):
    try:
        name = tif_name
        # print(name)
        tif_path = name
        png_target = tif_path
        img = Image.open(png_target)
        img = np.asarray(img)
        channel = color_channel[name.split('_')[-1][:-4]]
        if len(img.shape)==3:
            img = img[..., channel]
        # imageio.imwrite(tif_path, img)
        cv2.imwrite(tif_path.replace('HPAdata', 'data'), img)
        return
    except OSError as reason:
        print(f'file is not exist:{name}', '\nerror is', str(reason))
        # os.remove(png_target)


# target = r'.\publichpa_multilabel'
# target = '/home/xlzhu/2021_Kaggle_Competition/hpa_data_gray/publichpa_single'
# target = '/home/xlzhu/2021_Kaggle_Competition/hpa_data_gray/publichpa_multilabel_8byte'
# raw = '/home/xlzhu/2021_Kaggle_Competition/HPA_multilabel/HPA_multilabel_all/publichpa_multilabel_8byte'

def main():
    df = '/home/xlzhu/Work1_SingleCellPrediciton/meta/Extra_meta.csv'
    df = pd.read_csv(df)


    imgpath = '../HPA_data/IF/HPAdata'
    data_list = [join(imgpath, ID + '_' + color + '.png') for ID in df.ID.tolist() for color in color_channel.keys()]
    for i in data_list:
        rgb_to_gray(i)

    # pool = mlc.SuperPool(8)
    # pool.map(rgb_to_gray, data_list, description='convert hpa v20')


if __name__ == '__main__':
    main()
