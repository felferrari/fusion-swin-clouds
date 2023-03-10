from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
import numpy as np
import os
from conf import paths, general
import random
from skimage.util import view_as_windows
from utils.ops import load_sb_image, load_opt_image, load_SAR_image

class TrainDataSet(Dataset):
    def __init__(self, ds_prefix, device, year, data_aug = False, transformer = ToTensor()) -> None:
        self.device = device
        self.data_aug = data_aug
        self.transformer = transformer

        self.year_0 = str(year-1)[2:]
        self.year_1 = str(year)[2:]

        opt_files = os.listdir(paths.PREPARED_OPT_PATH)
        sar_files = os.listdir(paths.PREPARED_SAR_PATH)

        opt_files_0 = [os.path.join(paths.PREPARED_OPT_PATH, fi) for fi in opt_files if fi.startswith(self.year_0)]
        opt_files_1 = [os.path.join(paths.PREPARED_OPT_PATH, fi) for fi in opt_files if fi.startswith(self.year_1)]

        sar_files_0 = [os.path.join(paths.PREPARED_SAR_PATH, fi) for fi in sar_files if fi.startswith(self.year_0)]
        sar_files_1 = [os.path.join(paths.PREPARED_SAR_PATH, fi) for fi in sar_files if fi.startswith(self.year_1)]

        label = load_sb_image(os.path.join(paths.GENERAL_PATH, f'{general.LABEL_PREFIX}_{year}.tif'))
        self.shape = label.shape

        prev_def = load_sb_image(os.path.join(paths.GENERAL_PATH, f'{general.PREVIOUS_PREFIX}_{year}.tif'))

        self.idx_patches = np.load(os.path.join(paths.PREPARED_GENERAL_PATH, f'{ds_prefix}_{year}.npy'))

        self.opt_imgs_0 = [np.load(img_path).reshape(-1, general.N_OPTICAL_BANDS) for img_path in opt_files_0]
        self.opt_imgs_1 = [np.load(img_path).reshape(-1, general.N_OPTICAL_BANDS) for img_path in opt_files_1]

        self.sar_imgs_0 = [np.load(img_path).reshape(-1, general.N_SAR_BANDS) for img_path in sar_files_0]
        self.sar_imgs_1 = [np.load(img_path).reshape(-1, general.N_SAR_BANDS) for img_path in sar_files_1]

        self.label = label.flatten()
        self.prev_def = prev_def.reshape(-1,1)

    def __len__(self):
        return self.idx_patches.shape[0] * general.N_IMAGES_YEAR**2


    def __getitem__(self, index):
        idx_patch = index // (general.N_IMAGES_YEAR**2)
        im_0 = (index % general.N_IMAGES_YEAR) // general.N_IMAGES_YEAR
        im_1 = (index % general.N_IMAGES_YEAR) %  general.N_IMAGES_YEAR

        patch = self.idx_patches[idx_patch]

        if self.data_aug:
            k = random.randint(0, 3)
            patch = np.rot90(patch)

            if bool(random.getrandbits(1)):
                patch = np.flip(patch, axis=0)

            if bool(random.getrandbits(1)):
                patch = np.flip(patch, axis=1)

        opt_0 = self.transformer(self.opt_imgs_0[im_0][patch].astype(np.float32)).to(self.device)
        opt_1 = self.transformer(self.opt_imgs_1[im_1][patch].astype(np.float32)).to(self.device)

        sar_0 = self.transformer(self.sar_imgs_0[im_0][patch].astype(np.float32)).to(self.device)
        sar_1 = self.transformer(self.sar_imgs_1[im_1][patch].astype(np.float32)).to(self.device)

        prev_def = self.transformer(self.prev_def[patch].astype(np.float32)).to(self.device)
        label = torch.tensor(self.label[patch].astype(np.int64)).to(self.device)

        return (
            opt_0,
            opt_1,
            sar_0,
            sar_1,
            prev_def
        ), label

class PredDataSet(Dataset):
    def __init__(self, device, year, img_pair, transformer = ToTensor()) -> None:
        self.device = device
        self.transformer = transformer

        self.year_0 = str(year-1)[2:]
        self.year_1 = str(year)[2:]

        opt_files = os.listdir(paths.PREPARED_OPT_PATH)
        sar_files = os.listdir(paths.PREPARED_SAR_PATH)

        opt_files_0 = [os.path.join(paths.PREPARED_OPT_PATH, fi) for fi in opt_files if fi.startswith(self.year_0)]
        opt_files_1 = [os.path.join(paths.PREPARED_OPT_PATH, fi) for fi in opt_files if fi.startswith(self.year_1)]

        sar_files_0 = [os.path.join(paths.PREPARED_SAR_PATH, fi) for fi in sar_files if fi.startswith(self.year_0)]
        sar_files_1 = [os.path.join(paths.PREPARED_SAR_PATH, fi) for fi in sar_files if fi.startswith(self.year_1)]

        pad_shape = ((general.PATCH_SIZE, general.PATCH_SIZE),(general.PATCH_SIZE, general.PATCH_SIZE))

        self.prev_def_file = os.path.join(paths.GENERAL_PATH, f'{general.PREVIOUS_PREFIX}_{year}.tif')
        prev_def = load_sb_image(self.prev_def_file)
        self.original_shape = prev_def.shape
        prev_def = np.pad(prev_def, pad_shape, mode = 'reflect')
        self.padded_shape = prev_def.shape[:2]
        self.prev_def = prev_def.reshape((-1, 1))

        pad_shape = ((general.PATCH_SIZE, general.PATCH_SIZE),(general.PATCH_SIZE, general.PATCH_SIZE),(0,0))

        self.opt_file_0 = opt_files_0[img_pair[0]]
        self.opt_file_1 = opt_files_1[img_pair[1]]

        img = np.load(self.opt_file_0)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.opt_img_0 = img.reshape((-1, img.shape[-1]))

        img = np.load(self.opt_file_1)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.opt_img_1 = img.reshape((-1, img.shape[-1]))
        
        self.sar_file_0 = sar_files_0[img_pair[0]]
        self.sar_file_1 = sar_files_1[img_pair[1]]

        img = np.load(self.sar_file_0)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.sar_img_0 = img.reshape((-1, img.shape[-1]))

        img = np.load(self.sar_file_1)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.sar_img_1 = img.reshape((-1, img.shape[-1]))

        self.label = load_sb_image(os.path.join(paths.GENERAL_PATH, f'{general.LABEL_PREFIX}_{year}.tif'))


    def gen_patches(self, overlap):
        idx_patches = np.arange(self.padded_shape[0]*self.padded_shape[1]).reshape(self.padded_shape)
        slide_step = int((1-overlap)*general.PATCH_SIZE)
        window_shape = (general.PATCH_SIZE, general.PATCH_SIZE)
        self.idx_patches = view_as_windows(idx_patches, window_shape, slide_step).reshape((-1, general.PATCH_SIZE, general.PATCH_SIZE))

    def __len__(self):
        return self.idx_patches.shape[0]

    def __getitem__(self, index):
        patch = self.idx_patches[index]

        opt_0 = self.transformer(self.opt_img_0[patch].astype(np.float32)).to(self.device)
        opt_1 = self.transformer(self.opt_img_1[patch].astype(np.float32)).to(self.device)

        sar_0 = self.transformer(self.sar_img_0[patch].astype(np.float32)).to(self.device)
        sar_1 = self.transformer(self.sar_img_1[patch].astype(np.float32)).to(self.device)

        prev_def = self.transformer(self.prev_def[patch].astype(np.float32)).to(self.device)

        return (
            opt_0,
            opt_1,
            sar_0,
            sar_1,
            prev_def
        )