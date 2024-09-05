"""
Sanity checks for dino inference (plot heatmaps)
"""
import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import yaml
import os
import numpy as np

from utils import half_gaussian_filter_1d, rewards_matrix_heatmap, rewards_line_plot, pad_to_longest_sequence
from hiera_model import HieraRewardModel
import hydra

all_source_thresh = [0, .001]


def load_frames_to_pil(gif_path):
    gif_obj = Image.open(gif_path)
    frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]
    
    
    # many of the gifs have the last frame standing, which will mess with smoothing
    # chop off last one 
    return frames[:-1]
    
def load_frames_to_torch(gif_path):
    frames = load_frames_to_pil(gif_path)
    np_frames = np.stack([np.array(frame) for frame in frames])

    return  torch_frames 

def load_frames_to_numpy(gif_path):
    frames = load_frames_to_pil(gif_path)
    np_frames = np.stack([pil_to_tensor(frame) for frame in frames])

    return  np_frames 


def rewards_from_gifs(model, reward_transform, gif_paths):
        
    all_frames = []
    for gif_path in gif_paths:
        frames = load_frames_to_numpy(gif_path)
        all_frames.append(frames)    

    all_rewards = []

    for frames in all_frames:
        emb = model.set_source_embeddings(np.transpose(frames, (0, 2, 3, 1))) # convert to bhwc, as expected by SAM2
        rewards = model.predict()
        transformed_rewards = reward_transform(rewards)
        
        all_rewards.append(transformed_rewards)
    
    all_rewards = pad_to_longest_sequence(all_rewards)
    all_rewards = np.stack(all_rewards)
    n_gifs = len(gif_paths)
    labels = [gif_path.split('/')[-1].split('.')[0] for gif_path in gif_paths]

    return all_rewards, labels


def experiment_2():

    gif_paths =  ['../../debugging/gifs/kneeling_gifs/decent_kneeling.gif',
                '../../debugging/gifs/kneeling_gifs/kneel_adversary.gif',
                '../../debugging/gifs/kneeling_gifs/leaning_forward.gif',
                '../../debugging/gifs/kneeling_gifs/crossq_kneel.gif',
                '../../debugging/gifs/standing_gifs/crossq_stand.gif'
               ]

    def smooth_transform(rewards):
        return half_gaussian_filter_1d(rewards, sigma=4, smooth_last_N=True) 

    sam2_cfg_path = "sam2_hiera_b+.yaml"
    sam2_checkpoint_path = "./checkpoints/sam2_hiera_base_plus.pt"
    
    model = HieraRewardModel(sam2_cfg_path, sam2_checkpoint_path)


    target = np.array(Image.open("/share/portal/wph52/CrossQ/sbx/vlm_reward/reward_models/language_irl/preference_data/mujoco_selected_images/kneeling/success/standup_p=4_0_frame_0034.jpeg").convert("RGB"))
    
    model.set_target_embedding(target)
    rewards, all_labels = rewards_from_gifs(model, smooth_transform, gif_paths) 

    rewards_matrix_heatmap(rewards, 'hiera.png')


if __name__=="__main__":
    experiment_2()
#    gif_paths = ['sbx/vlm_reward/reward_models/language_irl/kneeling_gifs_ranked/kneeling_5.gif']

    # for gif_path in gif_paths:
    #     rewards, all_labels = rewards_from_gifs([gif_path], 
    #                                 reward_config_dict=reward_config_dict, 
    #                                 reward_model_name=reward_model_name, 
    #                                 batch_size=batch_size, 
    #                                 sigma=sigma, 
    #                                 transform=identity)


    #     for i, rew in enumerate(rewards):
    #         colors = ['blue', 'green']
    #         fname = f"gif={gif_path.split('/')[-1]}_t={all_source_thresh[i]}"
    #         fp = os.path.join(base_save_path, fname)
    #         rewards_line_plot(rew, labels = [f"t={all_source_thresh[i]}"], fp=fp, c=colors[i % len(colors)])


     #rewards_matrix_heatmap(np.array(rewards), os.path.join(save_base, 'heatmap'))
    #rewards_matrix_heatmap(np.array(smoothed_rewards), os.path.join(save_base, 'heatmap_smooth'))
