import json
import numpy as np
import os
import sys
from osgeo import gdal_array
from osgeo import ogr, gdal, gdalconst

def load_json(fp):
    with open(fp) as f:
        return json.load(f)
    
def save_json(dict_, fp):
    with open(fp, 'w') as f:
        json.dump(dict_, f, indent=4)
    
    
def load_opt_image(patch):
    """load optical data.

    Args:
        patch (str): path to the geotiff optical file.

    Returns:
        array:numpy array of the image.
    """
    img = gdal_array.LoadFile(patch)
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1)

def load_sb_image(path):
    """load a single band geotiff image.

    Args:
        path (str): path to the geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(path)
    return img

def load_SAR_image(path):
    """load SAR image, converting from Db to DN.

    Args:
        path (str): path to the SAR geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    db_img = gdal_array.LoadFile(path)
    temp_dn_img = 10**(db_img/10)
    temp_dn_img[temp_dn_img>1] = 1
    return np.moveaxis(temp_dn_img, 0, -1)

def load_SAR_DN_image(path):
    '''Function to read SAR images'''
    im = gdal_array.LoadFile(path)
    return np.expand_dims(im, axis=-1)

def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
    return img

def create_exps_paths(exp_n):
    exps_path = 'exps'

    exp_path = os.path.join(exps_path, f'exp_{exp_n}')
    models_path = os.path.join(exp_path, 'models')

    results_path = os.path.join(exp_path, 'results')
    predictions_path = os.path.join(results_path, 'predictions')
    visual_path = os.path.join(results_path, 'visual')

    logs_path = os.path.join(exp_path, 'logs')

    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    return exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path

def load_exp(exp_n = None):
    if exp_n is None:
        if len(sys.argv)==1:
            return None
        else:
            return load_json(os.path.join('conf', 'exps', f'exp_{sys.argv[1]}.json'))
    else:
        return load_json(os.path.join('conf', 'exps', f'exp_{exp_n}.json'))
    

"""class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)"""

def save_geotiff(base_image_path, dest_path, data, dtype):
    """Save data array as geotiff.
    Args:
        base_image_path (str): Path to base geotiff image to recovery the projection parameters
        dest_path (str): Path to geotiff image
        data (array): Array to be used to generate the geotiff
        dtype (str): Data type of the destiny geotiff: If is 'byte' the data is uint8, if is 'float' the data is float32
    """
    base_data = gdal.Open(base_image_path, gdalconst.GA_ReadOnly)

    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()

    if len(data.shape) == 2:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Byte)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Float32)
            data = data.astype(np.float32)
    elif len(data.shape) == 3:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Byte)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Float32)
            data = data.astype(np.float32)
            
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetSpatialRef(crs)
    target_ds.SetProjection(proj)

    if len(data.shape) == 2:
        target_ds.GetRasterBand(1).WriteArray(data)
    elif len(data.shape) == 3:
        for band_i in range(1, data.shape[-1]+1):
            target_ds.GetRasterBand(band_i).WriteArray(data[:,:,band_i-1])
    target_ds = None

def count_parameters_old(model):
    """Count the number of model parameters
    Args:
        model (Module): Model
    Returns:
        int: Number of Model's parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    total_params = 0
    text = ''
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
        text += f'{name}: {params:,}\n'
    text+=f'Total: {total_params:,}\n'
    return text