import argparse
import pathlib
from conf import paths, general, default
import numpy as np
import os
from utils.ops import load_opt_image, load_SAR_image, load_sb_image
from skimage.util import view_as_windows
import sys
from tqdm import tqdm
import logging
parser = argparse.ArgumentParser(
    description='prepare the original files, generating .npy files to be used in the training/testing steps'
)

parser.add_argument( # Path to the optical images
    '--opt-path',
    type = pathlib.Path,
    default = paths.OPT_PATH,
    help = 'Path to the optical images'
)

parser.add_argument( # Path to the SAR images
    '--sar-path',
    type = pathlib.Path,
    default = paths.SAR_PATH,
    help = 'Path to the SAR images'
)

parser.add_argument( # List of years to be processed
    '-y', '--years',
    type = int,
    default = default.PROCESSED_YEARS,
    nargs='+',
    help = 'List of years to be prepared'
)

parser.add_argument( # The path to the experiments' folder
    '-x', '--experiments-folder',
    type = pathlib.Path,
    default = paths.EXPERIMENTS_PATH,
    help = 'The path to the experiments\' folder'
)

parser.add_argument( # The minimum proportion of deforestation labels in each train/validation patches
    '-d', '--def-min-prop',
    type =float,
    default = default.DEF_CLASS_MIN_PROP,
    help = 'The minimum proportion of deforestation labels in each train/validation patches'
)

args = parser.parse_args()

if not os.path.exists(paths.PREPARED_PATH):
    os.mkdir(paths.PREPARED_PATH)

np.random.seed(123)

outfile = os.path.join(args.experiments_folder, 'data-prep.txt')
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=outfile,
        filemode='w'
        )
log = logging.getLogger('preparing')

log.info('Generating statistics.')

opt_files = os.listdir(args.opt_path)
sar_files = os.listdir(args.sar_path)


opt_means, opt_stds = [], []
sar_means, sar_stds = [], []

for opt_file in opt_files:
    img = load_opt_image(os.path.join(args.opt_path, opt_file))
    img[np.isnan(img)] = 0
    opt_means.append(img.mean(axis=(0,1)))
    opt_stds.append(img.std(axis=(0,1)))

opt_mean = np.array(opt_means).mean(axis=0)
opt_std = np.array(opt_stds).mean(axis=0)

log.info(f'Optical mean: {opt_mean}')
log.info(f'Optical std: {opt_std}')

for sar_file in sar_files:
    img = load_opt_image(os.path.join(args.sar_path, sar_file))
    img[np.isnan(img)] = 0
    sar_means.append(img.mean(axis=(0,1)))
    sar_stds.append(img.std(axis=(0,1)))

sar_mean = np.array(sar_means).mean(axis=0)
sar_std = np.array(sar_stds).mean(axis=0)

log.info(f'SAR mean: {sar_mean}')
log.info(f'SAR std: {sar_std}')

log.info('Preparing files.')

for opt_file in opt_files:
    img = load_opt_image(os.path.join(args.opt_path, opt_file))
    img[np.isnan(img)] = 0
    img = (img - opt_mean)/opt_std
    log.info(f'Optical Image {opt_file} means: {img.mean(axis=(0,1))}')
    log.info(f'Optical Image {opt_file} stds: {img.std(axis=(0,1))}')
    np.save(os.path.join(paths.PREPARED_OPT_PATH, f'{opt_file[:-4]}'), img.astype(np.float16))

for sar_file in sar_files:
    img = load_opt_image(os.path.join(args.sar_path, sar_file))
    img[np.isnan(img)] = 0
    img = (img - sar_mean)/sar_std
    log.info(f'Optical Image {sar_file} means: {img.mean(axis=(0,1))}')
    log.info(f'Optical Image {sar_file} stds: {img.std(axis=(0,1))}')
    np.save(os.path.join(paths.PREPARED_SAR_PATH, f'{sar_file[:-4]}'), img.astype(np.float32))

log.info('Preparing patches.')
tiles = load_sb_image(paths.TILES_PATH).astype(np.uint8)
shape = tiles.shape
tiles = tiles.flatten()
idx = np.arange(shape[0] * shape[1]).reshape(shape)
window_shape = (general.PATCH_SIZE, general.PATCH_SIZE)
slide_step = int((1-general.OVERLAP_PROP)*general.PATCH_SIZE)
idx_patches = view_as_windows(idx, window_shape, slide_step).reshape((-1, general.PATCH_SIZE, general.PATCH_SIZE))
min_prop = args.def_min_prop

for year in tqdm(args.years[1:-1], desc = 'Preparing patches'):
    label = load_sb_image(os.path.join(paths.GENERAL_PATH, f'{general.LABEL_PREFIX}_{year}.tif')).astype(np.uint8).flatten()

    keep = ((label[idx_patches] == 1).sum(axis=(1,2)) / general.PATCH_SIZE**2) >= min_prop

    keep_args = np.argwhere(keep == True).flatten() #args with at least min_prop deforestation
    no_keep_args = np.argwhere(keep == False).flatten() #args with less than min_prop of deforestation
    no_keep_args = np.random.choice(no_keep_args, (keep==True).sum())

    keep_final = np.concatenate((keep_args, no_keep_args))

    all_idx_patches = idx_patches[keep_final]
    #all_idx_patches = idx_patches[keep_args]

    keep_val = (tiles[all_idx_patches] == 0).sum(axis=(1,2)) == general.PATCH_SIZE**2
    keep_train = (tiles[all_idx_patches] == 1).sum(axis=(1,2)) == general.PATCH_SIZE**2

    print(f'Train patches: {keep_train.sum()}')
    print(f'Validation patches: {keep_val.sum()}')

    val_idx_patches = all_idx_patches[keep_val]
    train_idx_patches = all_idx_patches[keep_train]

    np.save(os.path.join(paths.PREPARED_GENERAL_PATH, f'{general.VAL_PREFIX}_{year}'), val_idx_patches)
    np.save(os.path.join(paths.PREPARED_GENERAL_PATH, f'{general.TRAIN_PREFIX}_{year}'), train_idx_patches)
