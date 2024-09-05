import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image
from typing import Callable

def hiera_image_distance(wasserstein_f: Callable[[np.ndarray, np.ndarray], float], 
                            image_1: np.ndarray, image_2: np.ndarray):
    
    predictor.set_image_batch([image_1, image_2])

    d_high_res = wasserstein_f(predictor["high_res_feats"][0][0], predictor["high_res_feats"][0][1])
    d_mid_res = wasserstein_f(predictor["high_res_feats"][1][0], predictor["high_res_feats"][1][1])
    d_low_res = wasserstein_f(predictor["image_embed"][0], predictor["image_embed"][1])

    return d_low_res + d_mid_res + d_high_res

checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

image_1 = np.asarray(Image.open("/share/portal/wph52/CrossQ/debugging/images/mujoco_standing.png").convert('RGB'))
image_2 = np.asarray(Image.open("/share/portal/wph52/CrossQ/debugging/images/mujoco_standing.png").convert('RGB'))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image_batch([image_1, image_2])
