import numpy as np

import PIL
import skimage.exposure
from PIL import ImageEnhance

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def contrast_effect(array, strength: float, amplitude: float):
    """Sample effect: increase image contrast for single frame processing"""
    contrast_factor = 1 + amplitude * strength
    mean = np.mean(array)
    contrasted = (array - mean) * contrast_factor + mean
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def flash_effect(array, strength: float, amplitude: float):
    """Sample effect: increase image intensity for single frame processing"""
    intensity_factor = 255 - (amplitude * strength * 255)
    return skimage.exposure.rescale_intensity(array, (0, intensity_factor))


def contrast_effect_gpu(array, strength: float, amplitude: float):
    """MPS-native contrast effect optimized for single frame processing"""
    was_numpy = isinstance(array, np.ndarray)
    if was_numpy:
        device = torch.device("mps")
        tensor = torch.from_numpy(array).float().to(device)
    else:
        tensor = array.float()
        device = tensor.device
    
    contrast_factor = 1 + amplitude * strength
    mean = tensor.mean()
    contrasted = (tensor - mean) * contrast_factor + mean
    result = torch.clamp(contrasted, 0, 255)
    
    if was_numpy:
        return result.cpu().numpy().astype(np.uint8)
    else:
        return result.to(torch.uint8)


def flash_effect_gpu(array, strength: float, amplitude: float):
    """MPS-native flash effect optimized for single frame processing"""
    was_numpy = isinstance(array, np.ndarray)
    if was_numpy:
        device = torch.device("mps")
        tensor = torch.from_numpy(array).float().to(device)
    else:
        tensor = array.float()
        device = tensor.device
    
    intensity_factor = 255 - (amplitude * strength * 255)
    normalized = tensor / 255.0
    
    img_min = normalized.min()
    img_max = normalized.max()
    if img_max > img_min:
        rescaled = (normalized - img_min) / (img_max - img_min) * (intensity_factor / 255.0)
    else:
        rescaled = normalized
    
    result = torch.clamp(rescaled * 255.0, 0, 255)
    
    if was_numpy:
        return result.cpu().numpy().astype(np.uint8)
    else:
        return result.to(torch.uint8)
