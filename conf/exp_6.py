from conf import general
from models.models import SwinUnetEF

def get_model(log):
    log.info('Model EF SWINUnet')
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 1
    input_depth_1 = 2*general.N_SAR_BANDS 
    #log.info(f'Model size: {model_depths}')
    log.info(f'Input depth 0: {input_depth_0}, Input depth 1: {input_depth_1}')
    input_depth = input_depth_0 + input_depth_1
    model = SwinUnetEF(input_depth, general.N_CLASSES)

    return model