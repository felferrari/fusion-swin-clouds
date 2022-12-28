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
    def __init__(self, device, year, data_aug = False, transformer = ToTensor()) -> None:
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
        shape = label.shape

        prev_def = load_sb_image(os.path.join(paths.GENERAL_PATH, f'{general.PREVIOUS_PREFIX}_{year}.tif'))

        opt_imgs_0 = [np.load(img_path) for img_path in opt_files_0]
        opt_imgs_1 = [np.load(img_path) for img_path in opt_files_1]

        sar_imgs_0 = [np.load(img_path) for img_path in sar_files_0]
        sar_imgs_1 = [np.load(img_path) for img_path in sar_files_1]

        print()

        