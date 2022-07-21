
import pandas as pd; pd.options.mode.chained_assignment = None;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
# import torch

# Built In Imports
from collections import Counter
from datetime import datetime
import multiprocessing
from glob import glob
import warnings
import imageio
import urllib
import zipfile
import pickle
import random
import shutil
import string
import math
from tqdm import tqdm
import time
import gzip
import ast
import io
import os
import gc
import re
# import IPython

# Visualization Imports
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# import plotly.express as px
# import seaborn as sns
from PIL import Image
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
# import plotly
import PIL
import cv2




# PRESETS
LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative"]
INT_2_STR = {x:LBL_NAMES[x] for x in np.arange(19)}
INT_2_STR_LOWER = {k:v.lower().replace(" ", "_") for k,v in INT_2_STR.items()}
STR_2_INT_LOWER = {v:k for k,v in INT_2_STR_LOWER.items()}
STR_2_INT = {v:k for k,v in INT_2_STR.items()}


def binary_mask_to_ascii(mask, mask_val=1):
    """Converts a binary mask into OID challenge encoding ascii text."""
    mask = np.where(mask == mask_val, 1, 0).astype(np.bool)

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)
    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


def rle_encoding(img, mask_val=1):
    """
    Turns our masks into RLE encoding to easily store them
    and feed them into models later on
    https://en.wikipedia.org/wiki/Run-length_encoding

    Args:
        img (np.array): Segmentation array
        mask_val (int): Which value to use to create the RLE

    Returns:
        RLE string

    """
    dots = np.where(img.T.flatten() == mask_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return ' '.join([str(x) for x in run_lengths])


def rle_to_mask(rle_string, height, width):
    """ Convert RLE sttring into a binary mask

    Args:
        rle_string (rle_string): Run length encoding containing
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from

    Returns:
        Numpy array of the binary segmentation mask for a given cell
    """
    rows, cols = height, width
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    return img


def decode_img(img, img_size=(224, 224), testing=False):
    """TBD"""

    # convert the compressed string to a 3D uint8 tensor
    if not testing:
        # resize the image to the desired size
        img = tf.image.decode_png(img, channels=1)
        return tf.cast(tf.image.resize(img, img_size), tf.uint8)
    else:
        return tf.image.decode_png(img, channels=1)


def preprocess_path_ds(rp, gp, bp, yp, lbl, n_classes=19, img_size=(224, 224), combine=True, drop_yellow=True):
    """ TBD """

    ri = decode_img(tf.io.read_file(rp), img_size)
    gi = decode_img(tf.io.read_file(gp), img_size)
    bi = decode_img(tf.io.read_file(bp), img_size)
    yi = decode_img(tf.io.read_file(yp), img_size)

    if combine and drop_yellow:
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0]], axis=-1), tf.one_hot(lbl, n_classes, dtype=tf.uint8)
    elif combine:
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0], yi[..., 0]], axis=-1), tf.one_hot(lbl, n_classes,
                                                                                               dtype=tf.uint8)
    elif drop_yellow:
        return ri, gi, bi, tf.one_hot(lbl, n_classes, dtype=tf.uint8)
    else:
        return ri, gi, bi, yi, tf.one_hot(lbl, n_classes, dtype=tf.uint8)


def create_pred_col(row):
    """ Simple function to return the correct prediction string

    We will want the original public test dataframe submission when it is
    available. However, we will use the swapped inn submission dataframe
    when it is not.

    Args:
        row (pd.Series): A row in the dataframe

    Returns:
        The prediction string
    """
    if pd.isnull(row.PredictionString_y):
        return row.PredictionString_x
    else:
        return row.PredictionString_y


def load_image(img_id, img_dir, testing=False, only_public=False):
    """ Load An Image Using ID and Directory Path - Composes 4 Individual Images """
    if only_public:
        return_axis = -1
        clr_list = ["red", "green", "blue"]
    else:
        return_axis = 0
        clr_list = ["red", "green", "blue", "yellow"]

    if not testing:
        rgby = [
            np.asarray(Image.open(os.path.join(img_dir, img_id + f"_{c}.png")), np.uint8) \
            for c in ["red", "green", "blue", "yellow"]
        ]
        return np.stack(rgby, axis=-1)
    else:
        # This is for cellsegmentator
        return np.stack(
            [np.asarray(decode_img(tf.io.read_file(os.path.join(img_dir, img_id + f"_{c}.png")), testing=True),
                        np.uint8)[..., 0] \
             for c in clr_list], axis=return_axis,
        )


def plot_rgb(arr, figsize=(12, 12)):
    """ Plot 3 Channel Microscopy Image """
    plt.figure(figsize=figsize)
    plt.title(f"RGB Composite Image", fontweight="bold")
    plt.imshow(arr)
    plt.axis(False)
    plt.show()


def convert_rgby_to_rgb(arr):
    """ Convert a 4 channel (RGBY) image to a 3 channel RGB image.

    Advice From Competition Host/User: lnhtrang

    For annotation (by experts) and for the model, I guess we agree that individual
    channels with full range px values are better.
    In annotation, we toggled the channels.
    For visualization purpose only, you can try blending the channels.
    For example,
        - red = red + yellow
        - green = green + yellow/2
        - blue=blue.

    Args:
        arr (numpy array): The RGBY, 4 channel numpy array for a given image

    Returns:
        RGB Image
    """

    rgb_arr = np.zeros_like(arr[..., :-1])
    rgb_arr[..., 0] = arr[..., 0]
    rgb_arr[..., 1] = arr[..., 1] + arr[..., 3] / 2
    rgb_arr[..., 2] = arr[..., 2]

    return rgb_arr


def plot_ex(arr, figsize=(20,6), title=None, plot_merged=True, rgb_only=False):
    """ Plot 4 Channels Side by Side """
    if plot_merged and not rgb_only:
        n_images=5
    elif plot_merged and rgb_only:
        n_images=4
    elif not plot_merged and rgb_only:
        n_images=4
    else:
        n_images=3
    plt.figure(figsize=figsize)
    if type(title) == str:
        plt.suptitle(title, fontsize=20, fontweight="bold")

    for i, c in enumerate(["Red Channel – Microtubles", "Green Channel – Protein of Interest", "Blue - Nucleus", "Yellow – Endoplasmic Reticulum"]):
        if not rgb_only:
            ch_arr = np.zeros_like(arr[..., :-1])
        else:
            ch_arr = np.zeros_like(arr)
        if c in ["Red Channel – Microtubles", "Green Channel – Protein of Interest", "Blue - Nucleus"]:
            ch_arr[..., i] = arr[..., i]
        else:
            if rgb_only:
                continue
            ch_arr[..., 0] = arr[..., i]
            ch_arr[..., 1] = arr[..., i]
        plt.subplot(1,n_images,i+1)
        plt.title(f"{c.title()}", fontweight="bold")
        plt.imshow(ch_arr)
        plt.axis(False)

        if plot_merged:
            plt.subplot(1, n_images, n_images)

            if rgb_only:
                plt.title(f"Merged RGB", fontweight="bold")
                plt.imshow(arr)
            else:
                plt.title(f"Merged RGBY into RGB", fontweight="bold")
                plt.imshow(convert_rgby_to_rgb(arr))
            plt.axis(False)

        plt.tight_layout(rect=[0, 0.2, 1, 0.97])
        plt.show()


def flatten_list_of_lists(l_o_l, to_string=False):
    if not to_string:
        return [item for sublist in l_o_l for item in sublist]
    else:
        return [str(item) for sublist in l_o_l for item in sublist]


def create_segmentation_maps(list_of_image_lists, segmentator, batch_size=8):
    """ Function to generate segmentation maps using CellSegmentator tool

    Args:
        list_of_image_lists (list of lists):
            - [[micro-tubules(red)], [endoplasmic-reticulum(yellow)], [nucleus(blue)]]
        batch_size (int): Batch size to use in generating the segmentation masks

    Returns:
        List of lists containing RLEs for all the cells in all images
    """

    all_mask_rles = {}
    for i in tqdm(range(0, len(list_of_image_lists[0]), batch_size), total=len(list_of_image_lists[0]) // batch_size):

        # Get batch of images
        sub_images = [img_channel_list[i:i + batch_size] for img_channel_list in
                      list_of_image_lists]  # 0.000001 seconds

        # Do segmentation
        cell_segmentations = segmentator.pred_cells(sub_images)
        nuc_segmentations = segmentator.pred_nuclei(sub_images[2])

        # post-processing
        for j, path in enumerate(sub_images[0]):
            img_id = path.replace("_red.png", "").rsplit("/", 1)[1]
            nuc_mask, cell_mask = label_cell(nuc_segmentations[j], cell_segmentations[j])
            new_name = os.path.basename(path).replace('red', 'mask')
            all_mask_rles[img_id] = [rle_encoding(cell_mask, mask_val=k) for k in range(1, np.max(cell_mask) + 1)]
    return all_mask_rles


def get_img_list(img_dir, return_ids=False, sub_n=None):
    """ Get image list in the format expected by the CellSegmentator tool """
    if sub_n is None:
        sub_n=len(glob(img_dir + '/' + f'*_red.png'))
    if return_ids:
        images = [sorted(glob(img_dir + '/' + f'*_{c}.png'))[:sub_n] for c in ["red", "yellow", "blue"]]
        return [x.replace("_red.png", "").rsplit("/", 1)[1] for x in images[0]], images
    else:
        return [sorted(glob(img_dir + '/' + f'*_{c}.png'))[:sub_n] for c in ["red", "yellow", "blue"]]


def get_contour_bbox_from_rle(rle, width, height, return_mask=True, ):
    """ Get bbox of contour as `xmin ymin xmax ymax`

    Args:
        rle (rle_string): Run length encoding containing
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from

    Returns:
        Numpy array for a cell bounding box coordinates
    """
    mask = rle_to_mask(rle, height, width).copy()
    cnts = grab_contours(
        cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        ))
    x, y, w, h = cv2.boundingRect(cnts[0])

    if return_mask:
        return (x, y, x + w, y + h), mask
    else:
        return (x, y, x + w, y + h)


def get_contour_bbox_from_raw(raw_mask):
    """ Get bbox of contour as `xmin ymin xmax ymax`

    Args:
        raw_mask (nparray): Numpy array containing segmentation mask information

    Returns:
        Numpy array for a cell bounding box coordinates
    """
    cnts = grab_contours(
        cv2.findContours(
            raw_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        ))
    xywhs = [cv2.boundingRect(cnt) for cnt in cnts]
    xys = [(xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]) for xywh in xywhs]
    return sorted(xys, key=lambda x: (x[1], x[0]))


def pad_to_square(a):
    """ Pad an array `a` evenly until it is a square """
    if a.shape[1]>a.shape[0]: # pad height
        n_to_add = a.shape[1]-a.shape[0]
        top_pad = n_to_add//2
        bottom_pad = n_to_add-top_pad
        a = np.pad(a, [(top_pad, bottom_pad), (0, 0), (0, 0)], mode='constant')

    elif a.shape[0]>a.shape[1]: # pad width
        n_to_add = a.shape[0]-a.shape[1]
        left_pad = n_to_add//2
        right_pad = n_to_add-left_pad
        a = np.pad(a, [(0, 0), (left_pad, right_pad), (0, 0)], mode='constant')
    else:
        pass
    return a


def cut_out_cells(rgby, rles, resize_to=(256, 256), square_off=True, return_masks=False, from_raw=True):
    """ Cut out the cells as padded square images

    Args:
        rgby (np.array): 4 Channel image to be cut into tiles
        rles (list of RLE strings): List of run length encoding containing
            segmentation mask information
        resize_to (tuple of ints, optional): The square dimension to resize the image to
        square_off (bool, optional): Whether to pad the image to a square or not

    Returns:
        list of square arrays representing squared off cell images
    """
    w, h = rgby.shape[:2]
    contour_bboxes = [get_contour_bbox_from_rle(rle, w, h, return_mask=return_masks) for rle in rles]
    if return_masks:
        masks = [x[-1] for x in contour_bboxes]
        contour_bboxes = [x[:-1] for x in contour_bboxes]

    arrs = [rgby[bbox[1]:bbox[3], bbox[0]:bbox[2], ...] for bbox in contour_bboxes]
    if square_off:
        arrs = [pad_to_square(arr) for arr in arrs]

    if resize_to is not None:
        arrs = [
            cv2.resize(pad_to_square(arr).astype(np.float32),
                       resize_to,
                       interpolation=cv2.INTER_CUBIC) \
            for arr in arrs
        ]
    if return_masks:
        return arrs, masks
    else:
        return arrs


def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts













