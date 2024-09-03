
from jaxtyping import Float

from typing import List, Union
import torch 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import ot
from scipy.ndimage._filters import correlate1d, _gaussian_kernel1d

def half_gaussian_filter_1d(x, sigma, smooth_last_N=False, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """
    filters x using the half normal distribution, defined by sigma.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius

    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights[:(len(weights) - 1) // 2] = 0
    weights /= sum(weights)

    filtered = correlate1d(x, weights, axis, output, mode, cval, 0)

    if not smooth_last_N:
        filtered[..., -lw:] = x[..., -lw:]
        
    return filtered 

def compute_patchwise_wasserstein_plan(features: Float[np.ndarray, "d_src"], 
                                        target_features: Float[np.ndarray, "d_tgt"], 
                                        metric: str='cosine') -> Float[np.ndarray, "d_src d_tgt"]:
    """
    Compute the optimal transport plan from features to target features, 
    scaled such that it sums to 1 along the target axis.
    This essentially gives a probability distribution over the target features for each source feature.
    """

    M = cdist(target_features, features, metric=metric)

    if M.shape[1] == 0:
        print("Error: no features found. Distance is considered -1.  \
        If you are masking the source or target, consider decreasing the threshold")
        return -1

    wasser = ot.sinkhorn([], [], M, reg=1,log=False)

    # scale so that transport plan sums to 1 along target axis
    # ensures different source sequence lengths give same scaled outputs
    return wasser * wasser.shape[0] 
    

def compute_patchwise_wasserstein_distance(features: Float[np.ndarray, "d_src"], 
                                        target_features: Float[np.ndarray, "d_tgt"], 
                                            metric: str ='cosine') -> float:

    """
    Compute the cumulative optimal transport distance between a single set of
    features and target_features
    """
    M = cdist(target_features, features, metric=metric)

    if M.shape[1] == 0:
        print("Error: no features found. Distance is considered -1.  \
        If you are masking the source or target, consider decreasing the threshold")
        return -1

    return ot.sinkhorn2([], [], M, reg=1,log=False)



