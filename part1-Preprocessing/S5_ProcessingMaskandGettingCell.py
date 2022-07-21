import pandas as pd
from utils import *
import os
from batchgenerators.utilities.file_and_folder_operations import *
import cv2
from skimage import measure
from tqdm import tqdm


def fill_targets_train(row):
    row.Label = str(row.Label)

    row.Label = np.array(row.Label.split("|")).astype(np.int)
    # row.Label_idx = np.array(row.Label).astype(np.int)
    for num in row.Label:
        name = LBL_NAMES[int(num)]
        row.loc[name] = 1
    row.Label = '|'.join(np.sort(np.unique(row.Label)).astype(str).tolist())
    return row


def generate_meta(meta_dir, filename):
    df_path = join(meta_dir, filename + '.csv')
    label_df = pd.read_csv(df_path)
    for key in INT_2_STR_LOWER.keys():
        label_df[LBL_NAMES[key]] = 0
    meta_df = label_df.apply(fill_targets_train, axis=1)

    # filename = filename.replace(filename.split('_')[0], filename.split('_')[0] + 'cell')
    save_path = join(meta_dir, filename + '.csv')
    meta_df.to_csv(save_path, index=False)


def ProcessingFile2Onehot(meta_dir, files):
    for file in tqdm(files, total=len(files), desc='Processing file'):
        generate_meta(meta_dir, file)


def get_bbox(mask):
    coor = []
    for i in range(1, mask.max()+1):
        m = np.where(mask == i)
        xmin, ymin = np.min(m, axis=1)
        xmax, ymax = np.max(m, axis=1)
        # coor.append([xmin, ymin, xmax, ymax])
        coor.append([ymin, xmin, ymax+1, xmax+1])
    return sorted(coor, key=lambda x: (x[1], x[0], x[3], x[2]))


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


def get_cell_nuc_mask_images(meta_dir, file, save_dir, root_data, root_mask):
    def segmentingcellAndcalculateNume(itterows, TILE_SIZE=(224, 224)):
        ID = itterows['ID']
        segment_path = os.path.join(root_mask, ID + '_mask.png')
        mask = cv2.imread(segment_path, 0)

        ### Remove small regions that may be segmented incorrectly  ###
        region_area = []
        for i in range(1, mask.max() + 1):
            mask_ = np.where(mask == i, 1, 0)
            region_area.append(mask_.sum())
        region_area = np.array(region_area)
        move_area = np.where(region_area > np.median(region_area) * 0.15, 1, 0)
        if move_area.min() > 0:
            # print(ID)
            pass
        else:
            move_area = np.where(move_area == 0)[0]
            for i in move_area:
                mask[mask == i + 1] = 0
            mask = measure.label(mask)
            cv2.imwrite(segment_path, mask)

        ### Extract single-cell images from IF images, and resize to same pixel  ###
        batch_rgb_images = [
            cv2.imread(os.path.join(root_data, ID + f"_{c}.png"), 0) \
            for c in ["red", "green", "blue", "yellow"]
        ]
        # .convert('L')
        rgb_images = np.stack(batch_rgb_images, axis=-1)

        k = 0
        cell_tiles = []
        for _ in range(mask.max()):
            k = k + 1
            mask_cell = np.where(mask == k, 1, 0).astype(np.uint8)
            bbox = get_bbox(mask_cell)
            assert len(bbox) == 1
            bbox = bbox[0]
            rgb_image_only = rgb_images * np.stack([mask_cell] * 4, axis=-1)
            assert rgb_image_only.sum() != 0
            cell_tiles.append(cv2.resize(
                pad_to_square(
                    rgb_image_only[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]),
                TILE_SIZE, interpolation=cv2.INTER_CUBIC))
            assert cell_tiles[-1].sum() != 0

        maybe_mkdir_p(save_dir)
        save_label = join(save_dir, ID)
        # params = save_label, num, image_list['ID'], batch_cell_tiles[0][:, :, index]
        for i in range(mask.max()):
            save_label_color = save_label + f'_{i}.npy'
            np.save(save_label_color, cell_tiles[i])
        return mask.max()

    df = pd.read_csv(join(meta_dir, file + '.csv'))
    df['idx'] = 0
    tqdm.pandas(desc="segmenting cells")
    df['idx'] = df.progress_apply(lambda x: segmentingcellAndcalculateNume(x), axis=1)
    df['fold'] = 0
    save_path = join(meta_dir, file + '.csv')
    df.to_csv(save_path, index=False)


def SegmentingCellImageAndStatisticNum(meta_dir, files, save_dir, root_data, root_mask):
    for file in files:
        get_cell_nuc_mask_images(meta_dir, file, save_dir, root_data, root_mask)


def GettingCellFile(meta_dir, files, root_data, root_mask):
    def GettingInfoWidthAndNum(itterows):
        ID = itterows['ID']
        segment_path = os.path.join(root_mask, ID + '_mask.png')
        mask = cv2.imread(segment_path, 0)
        ImageWidth = mask.shape[-1]
        return ImageWidth

    for file in files:
        df = pd.read_csv(join(meta_dir, file + '.csv'))
        df['ImageWidth'] = 0
        tqdm.pandas(desc="Get image width")
        df['ImageWidth'] = df.progress_apply(lambda x: GettingInfoWidthAndNum(x), axis=1)
        save_path = join(meta_dir, file + '.csv')
        df.to_csv(save_path, index=False)

        ### Get .csv file with single cell index, add cell to the file name to distinguish it from the image ###
        print('\n[!] Get .csv file with single cell index')
        df['idx'] = df['idx'].apply(lambda x: list(map(lambda x:str(x), range(x))))
        df = df.explode('idx', ignore_index=True)
        df['ID'] = df['ID'].str.cat([df['idx']], sep='_')
        df = df[['ID', 'Label', 'fold', 'ImageWidth'] + LBL_NAMES]
        cellfile = file.replace(file.split('_')[0], file.split('_')[0] + 'cell')
        save_path = join(meta_dir, cellfile + '.csv')
        df.to_csv(save_path, index=False)


if __name__ == '__main__':

    print('%s: calling main function ... ' % os.path.basename(__file__))

    meta_dir = './csv/'
    root_mask = '../HPA_data/IF/mask'
    root_data = '../HPA_data/IF/data'
    save_dir = '../HPA_data/Cell/data'

    files = ['hpa_multi_bbox_meta', 'hpa_single_bbox_meta', 'train_multi_bbox_meta', 'train_single_bbox_meta']

    print('\n[!] Processing .csv file to onehot... ')
    ProcessingFile2Onehot(meta_dir, files)

    print('\n[!] Split cell images and add number of cells to each image')
    SegmentingCellImageAndStatisticNum(meta_dir, files, save_dir, root_data, root_mask)

    print('\n[!] Get .csv file with single cell index')
    GettingCellFile(meta_dir, files, root_data, root_mask)

    print('\nsuccess!')
