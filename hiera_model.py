import torch

import numpy as np
from PIL import Image

from jaxtyping import Float
from typing import Tuple, NewType, Callable

from model_interface import RewardModel
from utils import compute_patchwise_wasserstein_distance

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np


# hack to get hydra to work (if it's in the class, the config doesn't resolve)
sam2_cfg_path = "sam2_hiera_b+.yaml"
sam2_checkpoint_path = "./checkpoints/sam2_hiera_base_plus.pt"
predictor = SAM2ImagePredictor(build_sam2(sam2_cfg_path, sam2_checkpoint_path))

class HieraRewardModel(RewardModel):

    def __init__(self, sam2_cfg_path, sam2_checkpoint_path, metric='cosine'):
        self.predictor = predictor
        self.metric = metric

    def set_source_embeddings(self, image_batch: Float[np.ndarray, "b c h w"]):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

            self.predictor.set_image_batch(image_batch)

        self.high_res_sources = predictor._features["high_res_feats"][0].flatten(2).permute(0, 2, 1).cpu().float().numpy()
        self.mid_res_sources = predictor._features["high_res_feats"][1].flatten(2).permute(0, 2, 1).cpu().float().numpy()
        self.low_res_sources = predictor._features["image_embed"].cpu().flatten(2).permute(0, 2, 1).float().numpy()
        
    def set_target_embedding(self, target_image: Float[torch.Tensor, "c h w"]):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image_batch([target_image])

        self.high_res_target = predictor._features["high_res_feats"][0][0].flatten(1).T.cpu().float().numpy()
        self.mid_res_target = predictor._features["high_res_feats"][1][0].flatten(1).T.cpu().float().numpy()
        self.low_res_target= predictor._features["image_embed"][0].flatten(1).cpu().T.float().numpy()

    def predict(self) -> Float[torch.Tensor, "n"]:
        rewards = []


        for i in range(len(self.low_res_sources)):
            # high_res_d = compute_patchwise_wasserstein_distance(self.high_res_sources[i], self.high_res_target, self.metric)
            # mid_res_d = compute_patchwise_wasserstein_distance(self.mid_res_sources[i], self.mid_res_target, self.metric)
            low_res_d = compute_patchwise_wasserstein_distance(self.low_res_sources[i], self.low_res_target, self.metric)
            
            # average the rewards, so the range is still [0,1]
            # reward = (high_res_d + mid_res_d + low_res_d) / 3
            reward = low_res_d
            rewards.append(reward)

        return torch.as_tensor(rewards)

    def to(self, device):
        self.predictor.to(device)

    def cuda(self, rank:int):
        self.predictor.cuda(rank)

