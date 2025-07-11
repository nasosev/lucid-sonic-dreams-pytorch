import numpy as np
import random
import pickle
import requests
import json
import pandas as pd
import os
import logging

import librosa
import pygit2
import gdown
import torch

# from mega import Mega


def download_weights(url, output):
    """Download model weights from URL"""

    if "drive.google.com" in url:
        gdown.download(url, output=output, quiet=False)

    # elif "mega.nz" in url:
    #     m = Mega()
    #     m.login().download_url(url, dest_filename=output)

    elif "yadi.sk" in url:
        endpoint = (
            "https://cloud-api.yandex.net/v1/disk/"
            "public/resources/download?public_key="
        )
        r_pre = requests.get(endpoint + url)
        r_pre_href = r_pre.json().get("href")
        r = requests.get(r_pre_href)
        with open(output, "wb") as f:
            f.write(r.content)

    else:
        r = requests.get(url)
        with open(output, "wb") as f:
            f.write(r.content)


def consolidate_models():
    """Consolidate JSON dictionaries of pre-trained StyleGAN(2) weights"""

    # Define URL's for pre-trained StyleGAN and StyleGAN2 weights
    stylegan_url = (
        "https://raw.githubusercontent.com/justinpinkney/"
        "awesome-pretrained-stylegan/master/models.csv"
    )
    stylegan2_url = (
        "https://raw.githubusercontent.com/justinpinkney/"
        "awesome-pretrained-stylegan2/master/models.json"
    )

    # Load JSON dictionary of StyleGAN weights
    models_stylegan = pd.read_csv(stylegan_url).to_dict(orient="records")

    # Load JSON dictionary of StyleGAN2 weights
    r = requests.get(stylegan2_url)
    models_stylegan2 = json.loads(r.text)

    # Consolidate StyleGAN and StyleGAN2 weights
    all_models = models_stylegan + models_stylegan2

    return all_models


def get_spec_norm(wav, sr, n_mels, hop_length):
    """Obtain maximum value for each time-frame in Mel Spectrogram,
    and normalize between 0 and 1"""

    # Generate Mel Spectrogram
    spec_raw = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, hop_length=hop_length
    )

    # Obtain maximum value per time-frame
    spec_max = np.amax(spec_raw, axis=0)

    # Normalize all values between 0 and 1
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)

    return spec_norm


def interpolate(array_1: np.ndarray, array_2: np.ndarray, steps: int):
    """Linear interpolation between 2 arrays"""

    # Obtain evenly-spaced ratios between 0 and 1
    linspace = np.linspace(0, 1, steps)

    # Generate arrays for interpolation using ratios
    arrays = [(1 - l) * array_1 + (l) * array_2 for l in linspace]

    return np.asarray(arrays)


def full_frame_interpolation(frame_init, steps, len_output):
    """Given a list of arrays (frame_init), produce linear interpolations between
    each pair of arrays."""

    # Generate list of lists, where each inner list is a linear interpolation
    # sequence between two arrays.
    frames = [
        interpolate(frame_init[i], frame_init[i + 1], steps)
        for i in range(len(frame_init) - 1)
    ]

    # Flatten list of lists
    frames = [vec for interp in frames for vec in interp]

    # Repeat final vector until output is of the desired length
    while len(frames) < len_output:
        frames.append(frames[-1])

    return frames


def slerp(val, low, high):
    """Spherical linear interpolation. Supports both single vectors and batches."""
    # Handle batch operations
    if isinstance(val, (list, np.ndarray)) and np.array(val).ndim > 0:
        # Batch case: val is array of interpolation parameters
        val = np.array(val)
        result = np.zeros((len(val),) + low.shape)
        for i, v in enumerate(val):
            result[i] = slerp(v, low, high)
        return result
    
    # Single interpolation case (original behavior)
    low_norm = low / np.linalg.norm(low)
    high_norm = high / np.linalg.norm(high)
    dot = np.clip(np.dot(low_norm, high_norm), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.abs(omega) < 1e-10:
        return (1.0 - val) * low + val * high
    return (np.sin((1.0 - val) * omega) / np.sin(omega)) * low + (
        np.sin(val * omega) / np.sin(omega)
    ) * high


def generate_narrow_perturbation(latent_dim, scale=1e-4):
    return np.random.normal(loc=0.0, scale=scale, size=(latent_dim,)).astype(np.float32)


def constrain_noise(noise, center, max_radius):
    """Project `noise` back into a sphere of radius `max_radius` around `center`."""
    # Handle both single vectors and batches of vectors
    if noise.ndim == 1:
        # Single vector case (original behavior)
        diff = noise - center
        norm = np.linalg.norm(diff)
        if norm > max_radius:
            diff = diff / norm * max_radius
        return center + diff
    else:
        # Vectorized batch case
        diff = noise - center
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        # Only scale vectors that exceed max_radius
        scale = np.minimum(1.0, max_radius / np.maximum(norms, 1e-8))
        return center + diff * scale


def get_optimal_batch_size(model_input_shape: int = 512, 
                          target_memory_usage: float = 0.8,
                          min_batch_size: int = 1,
                          max_batch_size: int = 16) -> int:
    """
    Dynamically determine optimal batch size for Apple Silicon MPS.
    
    Args:
        model_input_shape: Input dimension for StyleGAN (typically 512)
        target_memory_usage: Target memory utilization (0.8 = 80%)
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size for stability
        
    Returns:
        Optimal batch size for Apple Silicon system
    """
    
    try:
        # Try to get system memory using psutil if available
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback: estimate based on common Apple Silicon configurations
            # This is a reasonable approximation for most systems
            import subprocess
            try:
                # Use system_profiler to get memory info on macOS
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True, timeout=5)
                if 'Memory:' in result.stdout:
                    # Parse memory from system_profiler output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Memory:' in line:
                            memory_str = line.split(':')[1].strip()
                            if 'GB' in memory_str:
                                total_memory_gb = float(memory_str.split()[0])
                                break
                    else:
                        total_memory_gb = 16.0  # Conservative default
                else:
                    total_memory_gb = 16.0  # Conservative default
            except Exception:
                total_memory_gb = 16.0  # Conservative default
        
        # Estimate available MPS memory based on system memory
        # Apple Silicon M1/M2/M3 use unified memory architecture
        # Typically allocate 60-80% of system memory to MPS
        estimated_mps_memory_gb = total_memory_gb * 0.7
        
        # Estimate memory per batch for Apple Silicon:
        # - StyleGAN inference: ~500MB per batch for 512x512
        # - Additional overhead: ~200MB
        memory_per_batch_gb = 0.7
        
        # Calculate optimal batch size
        max_batches_by_memory = int(estimated_mps_memory_gb * target_memory_usage / memory_per_batch_gb)
        optimal_batch_size = min(max(max_batches_by_memory, min_batch_size), max_batch_size)
        
        logging.info(f"System memory: {total_memory_gb:.1f}GB, "
                    f"estimated MPS memory: {estimated_mps_memory_gb:.1f}GB, "
                    f"optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
        
    except Exception as e:
        logging.warning(f"Failed to determine optimal batch size: {e}")
        return min_batch_size


def get_optimal_worker_count() -> int:
    """
    Determine optimal number of workers for data loading and concurrent processing.
    
    Returns:
        Optimal worker count based on system capabilities
    """
    try:
        # Get CPU count
        cpu_count = os.cpu_count() or 1
        
        # For I/O bound tasks (image saving), use more workers
        # For CPU bound tasks, use fewer to avoid context switching overhead
        # Conservative approach: use 75% of available cores, minimum 2, maximum 8
        optimal_workers = min(max(int(cpu_count * 0.75), 2), 8)
        
        logging.info(f"System has {cpu_count} CPUs, using {optimal_workers} workers")
        return optimal_workers
        
    except Exception as e:
        logging.warning(f"Failed to determine optimal worker count: {e}")
        return 4  # Fallback to reasonable default
