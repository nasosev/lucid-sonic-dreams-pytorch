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
    """Sample effect: increase image contrast using faster numpy operations
    
    Supports both single frame [H, W, C] and batch [B, H, W, C] operations
    """
    
    contrast_factor = 1 + amplitude * strength
    
    # Handle both single frame and batch processing
    if array.ndim == 4:  # Batch: [B, H, W, C]
        # Compute mean per image in batch
        mean = np.mean(array, axis=(1, 2, 3), keepdims=True)
    else:  # Single frame: [H, W, C]
        mean = np.mean(array)
    
    contrasted = (array - mean) * contrast_factor + mean
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def flash_effect(array, strength: float, amplitude: float):
    """Sample effect: increase image intensity
    
    Supports both single frame [H, W, C] and batch [B, H, W, C] operations
    """

    intensity_factor = 255 - (amplitude * strength * 255)
    
    # Handle both single frame and batch processing
    if array.ndim == 4:  # Batch: [B, H, W, C]
        # Apply rescale_intensity to each image in the batch
        result = np.zeros_like(array)
        for i in range(array.shape[0]):
            result[i] = skimage.exposure.rescale_intensity(array[i], (0, intensity_factor))
        return result
    else:  # Single frame: [H, W, C]
        return skimage.exposure.rescale_intensity(array, (0, intensity_factor))


def contrast_effect_gpu(array, strength: float, amplitude: float):
    """GPU-accelerated contrast effect using PyTorch
    
    Supports both single frame [H, W, C] and batch [B, H, W, C] operations
    Input can be numpy array or torch tensor
    """
    if not TORCH_AVAILABLE:
        return contrast_effect(array, strength, amplitude)
    
    # Convert to tensor if needed
    was_numpy = isinstance(array, np.ndarray)
    if was_numpy:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tensor = torch.from_numpy(array).float().to(device)
    else:
        tensor = array.float()
        device = tensor.device
    
    contrast_factor = 1 + amplitude * strength
    
    # Handle both single frame and batch processing
    if tensor.ndim == 4:  # Batch: [B, H, W, C]
        # Compute mean per image in batch
        mean = tensor.mean(dim=(1, 2, 3), keepdim=True)
    else:  # Single frame: [H, W, C]
        mean = tensor.mean()
    
    # Apply contrast adjustment
    contrasted = (tensor - mean) * contrast_factor + mean
    result = torch.clamp(contrasted, 0, 255)
    
    # Convert back to numpy if input was numpy
    if was_numpy:
        return result.cpu().numpy().astype(np.uint8)
    else:
        return result.to(torch.uint8)


def flash_effect_gpu(array, strength: float, amplitude: float):
    """GPU-accelerated flash effect using PyTorch
    
    Supports both single frame [H, W, C] and batch [B, H, W, C] operations
    Input can be numpy array or torch tensor
    """
    if not TORCH_AVAILABLE:
        return flash_effect(array, strength, amplitude)
    
    # Convert to tensor if needed
    was_numpy = isinstance(array, np.ndarray)
    if was_numpy:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tensor = torch.from_numpy(array).float().to(device)
    else:
        tensor = array.float()
        device = tensor.device
    
    intensity_factor = 255 - (amplitude * strength * 255)
    
    # Simple rescale operation that works for both single and batch
    # Normalize to 0-1, then rescale
    normalized = tensor / 255.0
    
    # Find min/max for rescaling
    if tensor.ndim == 4:  # Batch: [B, H, W, C]
        # Per-image min/max
        batch_size = tensor.shape[0]
        result_list = []
        for i in range(batch_size):
            img = normalized[i]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                rescaled = (img - img_min) / (img_max - img_min) * (intensity_factor / 255.0)
            else:
                rescaled = img
            result_list.append(rescaled)
        result = torch.stack(result_list) * 255.0
    else:  # Single frame: [H, W, C]
        img_min = normalized.min()
        img_max = normalized.max()
        if img_max > img_min:
            rescaled = (normalized - img_min) / (img_max - img_min) * (intensity_factor / 255.0)
        else:
            rescaled = normalized
        result = rescaled * 255.0
    
    result = torch.clamp(result, 0, 255)
    
    # Convert back to numpy if input was numpy
    if was_numpy:
        return result.cpu().numpy().astype(np.uint8)
    else:
        return result.to(torch.uint8)
