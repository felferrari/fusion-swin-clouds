from conf import general
from models.models import SwinUnetOpt

def get_model(log):
    log.info('Model Optical SWINUnet')
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 1
    input_depth_1 = 0
    #log.info(f'Model size: {model_depths}')
    log.info(f'Input depth 0: {input_depth_0}, Input depth 1: {input_depth_1}')
    input_depth = input_depth_0 + input_depth_1
    model = SwinUnetOpt(input_depth, general.N_CLASSES)

    return model