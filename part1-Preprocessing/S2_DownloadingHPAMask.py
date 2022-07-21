# -*- ecoding: utf-8 -*-

import io
import os
import cv2
import requests
import pathlib
import gzip
import imageio
import pandas as pd
import mlcrate as mlc
from PIL import Image
import PIL
import numpy as np
from skimage import measure
from batchgenerators.utilities.file_and_folder_operations import *
from skimage import filters,segmentation,measure,morphology, color
# from keras.utils.data_utils import get_file
import shutil


s = requests.session()
s.keep_alive = False

# All label names in the public HPA and their corresponding index.
all_locations = dict({
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18
})


def tif_gzip_to_png(tif_path):
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. for working in local work station
    '''
    try:
        name = tif_path
        # print(name)
        tif_path = join(cache_dir, name)
        png_target = join(cache_dir, 'data', name)
        png_path = pathlib.Path(png_target.replace('.tif.gz', '.png'))
        tf = gzip.open(tif_path).read()
        img = imageio.imread(tf, 'tiff')
        imageio.imwrite(png_path, img)

    except:
        print('error file %s , has been moving' % name)


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
    print(url)
    if not os.path.exists(target_path):
        i = 0
        while i < 5:
            try:
                r = requests.get(url, allow_redirects=True)
                open(target_path, 'wb').write(r.content)
                return
            except requests.exceptions.RequestException:
                i += 1


def convert_imgae_to_labelimgae(params):
    '''Function to convert .tif.gz to .png and put it in the same folder
    Eg. in Kaggle notebook
    '''
    url0, target_path = params
    if not os.path.exists(target_path):
        download_and_convert_jpg_to_png(params)
        url = url0.split('proteinatlas.org')[0]+'images.proteinatlas.org'+url0.split('proteinatlas.org')[-1]
    if not os.path.exists(target_path.replace('raw', 'mask')):
        url = url0.split('proteinatlas.org')[0] + 'images.proteinatlas.org' + url0.split('proteinatlas.org')[-1]
        try:
            if os.path.exists(target_path):
                img = Image.open(target_path)
                img = img.convert('L')
                img = np.asarray(img, np.uint8)
                assert img.shape[0] == img.shape[1], f'{img.shape[0]}!={img.shape[1]}'
                thresh = filters.threshold_otsu(img)
                bw = morphology.opening(img > thresh, morphology.disk(3))
                cleared = bw.copy()  # copy
                segmentation.clear_border(cleared)  # clear up the noise
                img = morphology.remove_small_objects(cleared, 1000)
                labels_imgs = measure.label(img)
                mask = np.array(labels_imgs > 0, np.uint8)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # for i in range(len(contours)):
                #     cv2.fillConvexPoly(mask, contours[i], 1)
                cv2.fillPoly(mask, contours, 1)
                mask = measure.label(mask)
                cv2.imwrite(target_path.replace('raw', 'mask'), mask)
                # imageio.imwrite(target_path.replace('raw', 'label'), labels_imgs)
                print('Successful:', target_path)
        except PIL.UnidentifiedImageError:
            colors = ['blue', 'red', 'green', 'yellow']
            red_path = target_path.replace('raw', 'data')
            red_path = red_path.replace('mask', 'red')
            blue_path = target_path.replace('raw', 'data')
            blue_path = blue_path.replace('mask', 'blue')
            yellow_path = target_path.replace('raw', 'data')
            yellow_path = yellow_path.replace('mask', 'yellow')
            green_path = target_path.replace('raw', 'data')
            green_path = green_path.replace('mask', 'green')
            for color in colors:
                try:
                    Image.open(eval(f'{color}_path'))
                except:
                    url0 = url.replace('/images_cell_segmentation', '')
                    url0 = url0.replace('png', 'tif.gz')
                    url0 = url0.replace('segmentation', f'{color}')
                    print('Downloading {}'.format(url0))
                    if not os.path.exists(join(cache_dir,url0.split('/')[-1])):
                        get_file(fname=url0.split('/')[-1],
                                 origin=url0, cache_subdir=cache_dir)
    else:
        try:
            img = Image.open(target_path)
        except:
            img = Image.open(target_path)
            img = img.convert('L')
            img = np.asarray(img, np.uint8)
            assert img.shape[0] == img.shape[1], f'{img.shape[0]}!={img.shape[1]}'
            thresh = filters.threshold_otsu(img)
            bw = morphology.opening(img > thresh, morphology.disk(3))
            cleared = bw.copy()  # copy
            segmentation.clear_border(cleared)  # clear up the noise

            img = morphology.remove_small_objects(cleared, 1000)
            labels_imgs = measure.label(img)
            cv2.imwrite(target_path, labels_imgs)
            # imageio.imwrite(target_path.replace('raw', 'label'), labels_imgs)
            print('error in :',target_path)


def png_to_256(tif_name):
    try:
        name = tif_name
        # print(name)
        target = join(cache_dir, 'data')
        tif_path = join(save_dir.replace('raw', 'data'), name)
        if not os.path.exists(target):
            os.makedirs(target)
        if os.path.exists(tif_path):
            return
        png_target = join(target, name)
        img = Image.open(png_target)
        img = np.asarray(img)
        s = img.dtype
        if s == 'uint8':
            shutil.copy(png_target, tif_path)
            return
        if s == 'int32':
            img = np.ceil(img * 255.0 / 2**16)
            imageio.imwrite(tif_path, img.astype(np.uint8))
            return
        if s == 'uint16':
            img = np.ceil(img * 255.0 / 2**16)
            imageio.imwrite(tif_path, img.astype(np.uint8))
            return
        print(name)
    except:
        print(name+'-error')


### The downloading mask directory can be modified here ###

cache_dir = join('../HPA_data/IF/', 'cache_gz')
maybe_mkdir_p(cache_dir)
save_dir = os.path.join('../HPA_data/IF/raw')


def main():

    pool = mlc.SuperPool(1)
    public_hpa_df = pd.read_csv('./csv/kaggle_2021.tsv')
    colors = ['segmentation']


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_dir.replace('raw', 'mask')):
        os.makedirs(save_dir.replace('raw', 'mask'))

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    params = []

    for img in public_hpa_df['Image'].values:
        for color in colors:
            img = img.split('org')
            img = img[0]+'org'+'/images_cell_segmentation'+img[1]
            img = img.replace('images.', '')
            img_url = f'{img}_{color}.png'
            save_path = os.path.join(save_dir, f'{os.path.basename(img)}_mask.png')
            params.append((img_url, save_path))

    pool.map(convert_imgae_to_labelimgae, params, description='convert hpa v20')

    data_list = subfiles(cache_dir, join=False, suffix='.gz')
    pool.map(tif_gzip_to_png, data_list, description='converting gz to png')

    data_list = listdir(join(cache_dir, 'data'))
    pool.map(png_to_256, data_list, description='converting gz to png')


if __name__ == '__main__':
    main()


