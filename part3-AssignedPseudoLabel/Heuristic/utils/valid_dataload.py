from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import torch
from utils.util import *
from batchgenerators.utilities.file_and_folder_operations import *
import torchvision
from skimage.measure import label


TILE_SIZE = (224, 224)



def get_bbox(mask):
    coor = []
    for i in range(1, mask.max()+1):
        m = np.where(mask == i)
        xmin, ymin = np.min(m, axis=1)
        xmax, ymax = np.max(m, axis=1)
        # coor.append([xmin, ymin, xmax, ymax])
        coor.append([ymin, xmin, ymax+1, xmax+1])
    return sorted(coor, key=lambda x: (x[1], x[0], x[3], x[2]))


class TestDataset(Dataset):
    def __init__(self, df, hpa_mean, hpa_std, root_mask):
        self.df = df
        self.IDs = df.ID.unique().tolist()
        self.root_mask = root_mask
        self.tensor_tfms = Compose([
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
            Normalize(mean=hpa_mean, std=hpa_std),
        ])

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        cell_num = len(self.df[self.df.ID.isin([ID])])
        batch_rgby_images = [
            load_image(ID, TEST_IMG_DIR, testing=False)
        ]
        #  获得使用的mask
        batch_masks = [cv2.imread(join(self.root_mask, ID + '_mask.png'), 0)]

        batch_rgb_images = [rgby_image for rgby_image in batch_rgby_images]

        # Get Bounding Boxes For All Cells in All Images in Batch
        # batch_cell_bboxes = [get_contour_bbox_from_raw(mask) for mask in batch_masks]
        batch_cell_bboxes = [get_bbox(mask) for mask in batch_masks]
        assert cell_num == len(batch_cell_bboxes[0]), print('%s,cell num is not match,%d!==%d' %
                                                                 (ID, cell_num, len(batch_cell_bboxes[0])))
        batch_cell_tiles = []
        index = 0
        for bboxes, rgb_image in zip(batch_cell_bboxes, batch_rgb_images):
            k = 0
            cell_tiles = []
            for bbox in bboxes:
                k = k + 1
                mask = np.where(batch_masks[index] == k, 1, 0).astype(np.uint8)
                rgb_image_only = rgb_image/255.0 * np.stack([mask] * 4, axis=-1)
                cell_tiles.append(cv2.resize(
                    pad_to_square(
                        rgb_image_only[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]),
                    TILE_SIZE, interpolation=cv2.INTER_CUBIC))
            batch_cell_tiles.append(cell_tiles)
            index = index + 1

        # Perform Inference
        batch_cell_tiles = [self.tensor_tfms(ct) for ct in batch_cell_tiles[0]]

        return torch.stack(batch_cell_tiles).float(), ID, len(batch_cell_tiles)


class TestDataset_ensemble(Dataset):
    def __init__(self, df, root_mask):
        self.df = df
        self.IDs = df.ID.unique().tolist()
        self.root_mask = root_mask
        self.tensor_tfms_img = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
            # Normalize(mean=[0.0994, 0.0466, 0.0606, 0.0879], std=[0.1406, 0.0724, 0.1541, 0.1264]),
            # Normalize(mean=hpa_mean, std=hpa_std),
        ])
        self.tensor_tfms_cell = Compose([
            ToTensor(),
            Normalize(mean=[0.0994, 0.0466, 0.0606, 0.0879], std=[0.1406, 0.0724, 0.1541, 0.1264]),
        ])

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        cell_num = len(self.df[self.df.ID.isin([ID])])
        batch_rgby_images = [
            load_image(ID, TEST_IMG_DIR, testing=False)
        ]
        #  获得使用的mask
        batch_masks = [cv2.imread(join(self.root_mask, ID + '_mask.png'), 0)]

        batch_rgb_images = [rgby_image for rgby_image in batch_rgby_images]

        # Get Bounding Boxes For All Cells in All Images in Batch
        # batch_cell_bboxes = [get_contour_bbox_from_raw(mask) for mask in batch_masks]
        batch_cell_bboxes = [get_bbox(mask) for mask in batch_masks]
        assert cell_num == len(batch_cell_bboxes[0]), print('%s,cell num is not match,%d!==%d' %
                                                                 (ID, cell_num, len(batch_cell_bboxes[0])))
        batch_cell_tiles = []
        index = 0
        for bboxes, rgb_image in zip(batch_cell_bboxes, batch_rgb_images):
            k = 0
            cell_tiles = []
            for bbox in bboxes:
                k = k + 1
                mask = np.where(batch_masks[index] == k, 1, 0).astype(np.uint8)
                rgb_image_only = rgb_image/255.0 * np.stack([mask] * 4, axis=-1)
                cell_tiles.append(cv2.resize(
                    pad_to_square(
                        rgb_image_only[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]),
                    TILE_SIZE, interpolation=cv2.INTER_CUBIC))
            batch_cell_tiles.append(cell_tiles)
            index = index + 1

        # Perform Inference
        batch_cell_tiles_img = [self.tensor_tfms_img(ct) for ct in batch_cell_tiles[0]]
        batch_cell_tiles_cell = [self.tensor_tfms_img(ct) for ct in batch_cell_tiles[0]]

        return torch.stack(batch_cell_tiles_img).float(), torch.stack(batch_cell_tiles_cell).float(), ID, len(batch_cell_tiles_img)


tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
            torchvision.transforms.Normalize(mean=[0.0994, 0.0466, 0.0606, 0.0879], std=[0.1406, 0.0724, 0.1541, 0.1264])
        ])


class HPADataSET(Dataset):
    def __init__(self, df, TEST_IMG_DIR, root_mask, tfms = tensor_tfms):
        self.df = df.reset_index(drop=True)
        self.tensor_tfms = tfms
        self.img_dir = TEST_IMG_DIR
        self.mask_dir = root_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        ID = self.df.loc[index, 'ID']
        rgby_image = load_image(ID, self.img_dir, testing=False)
        rgby_mask = cv2.imread(join(self.mask_dir, ID + '_mask.png'), 0)

        cell_bboxes = get_contour_bbox_from_raw(rgby_mask)

        rgby_mask = label(rgby_mask).copy()
        # cv2.imwrite(join('/home/Datasets/IF/Images/mask_', ID + '_mask.png'), rgby_mask)
        k = 0
        cell_tiles = []
        for bbox in cell_bboxes:
            k = k + 1
            mask = np.where(rgby_mask == k, 1, 0).astype(np.uint8)
            rgb_image_only = rgby_image * np.stack([mask] * 4, axis=-1)
            img = cv2.resize(pad_to_square(rgb_image_only[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]),
                TILE_SIZE, interpolation=cv2.INTER_CUBIC)
            img = self.tensor_tfms(img)
            cell_tiles.append(img)
        batch = torch.stack(cell_tiles, 0)
        return batch, index, k


