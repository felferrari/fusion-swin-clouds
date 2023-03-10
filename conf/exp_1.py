from conf import default, general, paths
from models.models import ResUnetOpt

def get_model(log):
    log.info('Model Optical Resunet')
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 1
    input_depth_1 = 0
    model_depths = [32, 64, 128, 256]
    log.info(f'Model size: {model_depths}')
    log.info(f'Input depth 0: {input_depth_0}, Input depth 1: {input_depth_1}')
    input_depth = input_depth_0 + input_depth_1
    model = ResUnetOpt(input_depth, model_depths, general.N_CLASSES)

    return model