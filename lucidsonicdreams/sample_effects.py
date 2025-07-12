import numpy as np

import PIL
import skimage.exposure
from PIL import ImageEnhance


def contrast_effect(array, strength: float, amplitude: float):
    """Sample effect: increase image contrast using faster numpy operations"""
    
    contrast_factor = 1 + amplitude * strength
    # Use numpy for faster contrast adjustment instead of PIL
    mean = np.mean(array)
    contrasted = (array - mean) * contrast_factor + mean
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def flash_effect(array, strength: float, amplitude: float):
    """Sample effect: increase image intensity"""

    intensity_factor = 255 - (amplitude * strength * 255)
    return skimage.exposure.rescale_intensity(array, (0, intensity_factor))
