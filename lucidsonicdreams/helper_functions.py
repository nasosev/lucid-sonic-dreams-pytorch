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
    and normalize using local windowing for better reactivity on long clips"""

    # Generate Mel Spectrogram
    spec_raw = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, hop_length=hop_length
    )

    # Obtain maximum value per time-frame
    spec_max = np.amax(spec_raw, axis=0)

    # Use local normalization with rolling window for better reactivity
    # Window size: ~10 seconds worth of frames for local context
    window_frames = max(1, int(10 * sr / hop_length))
    
    # Apply local normalization using rolling statistics
    spec_norm = np.zeros_like(spec_max)
    
    for i in range(len(spec_max)):
        # Define local window boundaries
        start_idx = max(0, i - window_frames // 2)
        end_idx = min(len(spec_max), i + window_frames // 2 + 1)
        
        # Get local window
        local_window = spec_max[start_idx:end_idx]
        
        # Local normalization with fallback for edge cases
        local_min = np.min(local_window)
        local_range = np.ptp(local_window)
        
        if local_range > 0:
            spec_norm[i] = (spec_max[i] - local_min) / local_range
        else:
            spec_norm[i] = 0.5  # Neutral value when no variation

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


def get_optimal_batch_size(
    model_input_shape: int = 512,
    target_memory_usage: float = 0.8,
    min_batch_size: int = 1,
    max_batch_size: int = 16,
) -> int:
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
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "Memory:" in result.stdout:
                    # Parse memory from system_profiler output
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "Memory:" in line:
                            memory_str = line.split(":")[1].strip()
                            if "GB" in memory_str:
                                total_memory_gb = float(memory_str.split()[0])
                                break
                    else:
                        total_memory_gb = 16.0  # Conservative default
                else:
                    total_memory_gb = 16.0  # Conservative default
            except Exception:
                total_memory_gb = 16.0  # Conservative default

        # Estimate available MPS memory based on system memory
        # Conservative approach for stability - avoid system crashes
        estimated_mps_memory_gb = total_memory_gb * 0.7

        # Estimate memory per batch for Apple Silicon:
        # - StyleGAN inference: ~500MB per batch for 512x512
        # - Additional overhead: ~200MB
        memory_per_batch_gb = 0.7

        # Calculate optimal batch size
        max_batches_by_memory = int(
            estimated_mps_memory_gb * target_memory_usage / memory_per_batch_gb
        )
        optimal_batch_size = min(
            max(max_batches_by_memory, min_batch_size), max_batch_size
        )

        logging.info(
            f"System memory: {total_memory_gb:.1f}GB, "
            f"estimated MPS memory: {estimated_mps_memory_gb:.1f}GB, "
            f"optimal batch size: {optimal_batch_size}"
        )

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


class OptimizedAudioProcessor:
    """
    Optimized audio processing class that eliminates redundant loading and FFT operations.
    Processes all audio channels efficiently with minimal memory overhead.
    """

    def __init__(self):
        self._audio_cache = {}
        self._hpss_cache = {}

    def load_and_process_audio(
        self,
        primary_audio: str,
        pulse_audio: str = None,
        motion_audio: str = None,
        class_audio: str = None,
        start: float = 0,
        duration: float = None,
        fps: int = 43,
        input_shape: int = 512,
        pulse_percussive: bool = True,
        pulse_harmonic: bool = False,
        motion_percussive: bool = False,
        motion_harmonic: bool = True,
    ) -> dict:
        """
        Load and process all audio files efficiently, eliminating redundant operations.

        Returns:
            Dictionary containing processed audio data and spectrograms
        """
        # Cache key for this specific audio configuration
        cache_key = f"{primary_audio}_{start}_{duration}_{fps}"

        # Load primary audio file once
        if cache_key not in self._audio_cache:
            wav_primary, sr_primary = librosa.load(
                primary_audio, offset=start, duration=duration
            )
            self._audio_cache[cache_key] = (wav_primary, sr_primary)
        else:
            wav_primary, sr_primary = self._audio_cache[cache_key]

        # Calculate frame duration once
        frame_duration = int(sr_primary / fps - (sr_primary / fps % 64))

        # Initialize audio channels
        audio_channels = {
            "pulse": {"wav": wav_primary, "sr": sr_primary},
            "motion": {"wav": wav_primary, "sr": sr_primary},
            "class": {"wav": wav_primary, "sr": sr_primary},
        }

        # Load separate audio files if provided (batch load)
        separate_files = []
        if pulse_audio and pulse_audio != primary_audio:
            separate_files.append(("pulse", pulse_audio))
        if motion_audio and motion_audio != primary_audio:
            separate_files.append(("motion", motion_audio))
        if class_audio and class_audio != primary_audio:
            separate_files.append(("class", class_audio))

        # Batch load separate files
        for channel, audio_file in separate_files:
            file_cache_key = f"{audio_file}_{start}_{duration}"
            if file_cache_key not in self._audio_cache:
                wav, sr = librosa.load(audio_file, offset=start, duration=duration)
                self._audio_cache[file_cache_key] = (wav, sr)
            else:
                wav, sr = self._audio_cache[file_cache_key]
            audio_channels[channel] = {"wav": wav, "sr": sr}

        # Perform harmonic/percussive separation only if needed
        hpss_needed = (not pulse_audio and pulse_percussive != pulse_harmonic) or (
            not motion_audio and motion_percussive != motion_harmonic
        )

        if hpss_needed:
            hpss_cache_key = f"{primary_audio}_{start}_{duration}_hpss"
            if hpss_cache_key not in self._hpss_cache:
                logging.info("Performing harmonic/percussive separation...")
                wav_harm, wav_perc = librosa.effects.hpss(wav_primary)
                self._hpss_cache[hpss_cache_key] = (wav_harm, wav_perc)
            else:
                wav_harm, wav_perc = self._hpss_cache[hpss_cache_key]

            # Assign optimal audio for pulse/motion based on preferences
            if not pulse_audio:
                if pulse_percussive and not pulse_harmonic:
                    audio_channels["pulse"]["wav"] = wav_perc
                elif pulse_harmonic and not pulse_percussive:
                    audio_channels["pulse"]["wav"] = wav_harm

            if not motion_audio:
                if motion_percussive and not motion_harmonic:
                    audio_channels["motion"]["wav"] = wav_perc
                elif motion_harmonic and not motion_percussive:
                    audio_channels["motion"]["wav"] = wav_harm

        # Batch compute spectrograms
        logging.info("Computing mel spectrograms...")
        spectrograms = {}
        for channel, audio_data in audio_channels.items():
            spectrograms[f"spec_norm_{channel}"] = get_spec_norm(
                audio_data["wav"], audio_data["sr"], input_shape, frame_duration
            )

        # Compute chromagram for class audio
        logging.info("Computing chromagram...")
        chrom_class = librosa.feature.chroma_cqt(
            y=audio_channels["class"]["wav"],
            sr=audio_channels["class"]["sr"],
            hop_length=frame_duration,
        )

        # Sort pitches based on dominance
        chrom_class_norm = chrom_class / chrom_class.sum(axis=0, keepdims=1)
        chrom_class_sum = np.sum(chrom_class_norm, axis=1)
        pitches_sorted = np.argsort(chrom_class_sum)[::-1]

        return {
            "spectrograms": spectrograms,
            "chrom_class": chrom_class,
            "pitches_sorted": pitches_sorted,
            "frame_duration": frame_duration,
            "primary_audio": (wav_primary, sr_primary),
        }
