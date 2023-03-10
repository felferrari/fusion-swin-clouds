from conf import general
from models.models import SwinUnetSAR

def get_model(log):
    log.info('Model SAR SWINUnet')
    input_depth_0 = 0
    input_depth_1 = 2*general.N_SAR_BANDS + 1
    #log.info(f'Model size: {model_depths}')
    log.info(f'Input depth 0: {input_depth_0}, Input depth 1: {input_depth_1}')
    input_depth = input_depth_0 + input_depth_1
    model = SwinUnetSAR(input_depth, general.N_CLASSES)

    return model