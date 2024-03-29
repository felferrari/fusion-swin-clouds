import argparse
import pathlib
import importlib
from conf import default, general, paths
import os
import time
import sys
from utils.dataloader import PredDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from utils.ops import save_geotiff
from multiprocessing import Process, freeze_support
from torch.multiprocessing import freeze_support
import logging

parser = argparse.ArgumentParser(
    description='Train NUMBER_MODELS models based in the same parameters'
)

parser.add_argument( # Experiment number
    '-e', '--experiment',
    type = int,
    default = 1,
    help = 'The number of the experiment'
)

parser.add_argument( # batch size
    '-b', '--batch-size',
    type = int,
    default = default.PREDICTION_BATCH_SIZE,
    help = 'The number of samples of each batch'
)

parser.add_argument( # Number of models to be trained
    '-n', '--number-models',
    type = int,
    default = default.N_MODELS,
    help = 'The number models to be trained from the scratch'
)

parser.add_argument( # Experiment path
    '-x', '--experiments-path',
    type = pathlib.Path,
    default = paths.EXPERIMENTS_PATH,
    help = 'The patch to data generated by all experiments'
)

parser.add_argument( # Reference year to predict
    '-y', '--year',
    type = int,
    default = default.PROCESSED_YEARS[2],
    help = 'Reference year to predict'
)

parser.add_argument( # base image 
    '-i', '--base-image-path',
    type = pathlib.Path,
    default = paths.OPT_PATH,
    help = 'Path to the folder of the geotiff files to generate aligned labels of the same region'
)

args = parser.parse_args()

exp_path = os.path.join(str(args.experiments_path), f'exp_{args.experiment}')
logs_path = os.path.join(exp_path, f'logs')
models_path = os.path.join(exp_path, f'models')
visual_path = os.path.join(exp_path, f'visual')
predicted_path = os.path.join(exp_path, f'predicted')
results_path = os.path.join(exp_path, f'results')


device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

#def run(model_idx):
#if __name__ == '__main__':
#    freeze_support()
#    for model_idx in tqdm(range(args.number_models), desc='Model Idx'):
    
def run(model_idx):

    outfile = os.path.join(logs_path, f'pred_{args.experiment}_{model_idx}.txt')
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=outfile,
            filemode='w'
            )
    log = logging.getLogger('predict')

    model_m =importlib.import_module(f'conf.exp_{args.experiment}')
    model = model_m.get_model(log)
    model.to(device)
    log.info(model)

    model_path = os.path.join(models_path, f'model_{model_idx}.pth')
    model.load_state_dict(torch.load(model_path))

    torch.set_num_threads(8)

    overlaps = general.PREDICTION_OVERLAPS
    log.info(f'Overlaps pred: {overlaps}')
    one_window = np.ones((general.PATCH_SIZE, general.PATCH_SIZE, general.N_CLASSES))
    total_time = 0
    n_images = 0

    for im_0 in tqdm(range(general.N_IMAGES_YEAR), leave=False, desc='Img 0'):
        for im_1 in tqdm(range(general.N_IMAGES_YEAR), leave = False, desc='Img 1'):
            img_pair = (im_0, im_1)
            dataset = PredDataSet(device = device, year = args.year, img_pair = img_pair)
            label = dataset.label
            log.info(f'Optical Image Year 0:{dataset.opt_file_0}')
            log.info(f'Optical Image Year 1:{dataset.opt_file_1}')
            log.info(f'SAR Image Year 0:{dataset.sar_file_0}')
            log.info(f'SAR Image Year 1:{dataset.sar_file_1}')
            #print(f'CMAP Image Year 0:{dataset.cmap_file_0}')
            #print(f'CMAP Image Year 1:{dataset.cmap_file_1}')
            log.info(f'Prev Def Image Year 1:{dataset.prev_def_file}')
            pred_global_sum = np.zeros(dataset.original_shape+(general.N_CLASSES,))
            t0 = time.perf_counter()
            for overlap in tqdm(overlaps, leave=False, desc='Overlap'):
                log.info(f'Predicting overlap {overlap}')
                dataset.gen_patches(overlap = overlap)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                
                pbar = tqdm(dataloader, desc='Prediction', leave = False)
                #preds = None
                preds = torch.zeros((len(dataset), general.N_CLASSES, general.PATCH_SIZE, general.PATCH_SIZE))
                for i, X in enumerate(pbar):
                    with torch.no_grad():
                        preds[args.batch_size*i: args.batch_size*(i+1)] =  model(X).to('cpu')
                preds = np.moveaxis(preds.numpy().astype(np.float16), 1, -1)
                pred_sum = np.zeros(dataset.padded_shape+(general.N_CLASSES,)).reshape((-1, general.N_CLASSES))
                pred_count = np.zeros(dataset.padded_shape+(general.N_CLASSES,)).reshape((-1, general.N_CLASSES))
                for idx, idx_patch in enumerate(tqdm(dataset.idx_patches, desc = 'Rebuild', leave = False)):
                    crop_val = general.PREDICTION_REMOVE_BORDER
                    idx_patch_crop = idx_patch[crop_val:-crop_val, crop_val:-crop_val]
                    pred_sum[idx_patch_crop] += preds[idx][crop_val:-crop_val, crop_val:-crop_val]
                    pred_count[idx_patch_crop] += one_window[crop_val:-crop_val, crop_val:-crop_val]

                    #pred_sum[idx_patch] += preds[idx]
                    #pred_count[idx_patch] += one_window
                pred_sum = pred_sum.reshape(dataset.padded_shape+(general.N_CLASSES,))
                pred_count = pred_count.reshape(dataset.padded_shape+(general.N_CLASSES,))

                pred_sum = pred_sum[general.PATCH_SIZE:-general.PATCH_SIZE,general.PATCH_SIZE:-general.PATCH_SIZE,:]
                pred_count = pred_count[general.PATCH_SIZE:-general.PATCH_SIZE,general.PATCH_SIZE:-general.PATCH_SIZE,:]

                pred_global_sum += pred_sum / pred_count

            p_time = (time.perf_counter() - t0)/60
            total_time += p_time
            n_images += 1
            log.info(f'Prediction time: {p_time} mins')
            pred_global = pred_global_sum / len(overlaps)
            #pred_b = pred_global.argmax(axis=-1).astype(np.uint8)

            #pred_b[label == 2] = 2

            #np.save(os.path.join(predicted_path, f'{general.PREDICTION_PREFIX}_{img_pair[0]}_{img_pair[1]}_{model_idx}.npy'), pred_b)
            np.save(os.path.join(predicted_path, f'{general.PREDICTION_PREFIX}_prob_{img_pair[0]}_{img_pair[1]}_{model_idx}.npy'), pred_global[:,:,1].astype(np.float16))

            #save_geotiff(str(args.base_image), os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_b, dtype = 'byte')

            pred_b2 = (pred_global[:,:,1] > 0.5).astype(np.uint8)
            pred_b2[label == 2] = 2
            base_image = os.listdir(str(args.base_image_path))[0]

            base_data = os.path.join(str(args.base_image_path), base_image)
            save_geotiff(base_data, os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_b2, dtype = 'byte')
            #save_geotiff(str(args.base_image), os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_probs_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_global, dtype = 'float')
    m_time = total_time / n_images
    log.info(f'Mean Prediction time: {m_time} mins')


if __name__=="__main__":
    
    for model_idx in range(args.number_models):
        print(f'Predicting model {model_idx}')
        p = Process(target=run, args=(model_idx,))
        p.start()
        p.join()
  