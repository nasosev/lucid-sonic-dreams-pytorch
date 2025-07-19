# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of Lucid Sonic Dreams that syncs StyleGAN-generated visuals to music. This fork focuses on Apple Silicon Metal Performance Shaders (MPS) support and StyleGAN3 integration, removing CUDA support. It generates dynamic video content by analyzing audio features and translating them into visual effects through neural networks.

## Key Architecture Components

### Core Classes and Modules

- **`LucidSonicDream`** (`lucidsonicdreams/main.py:71`): Main class orchestrating the entire pipeline from audio analysis to video generation
- **`EffectsGenerator`** (`lucidsonicdreams/main.py:1026`): Handles custom visual effects applied to generated frames
- **`BatchEffectsGenerator`** (`lucidsonicdreams/main.py:1083`): Optimized effects generator with vectorized batch processing
- **`GPUBatchEffectsGenerator`** (`lucidsonicdreams/main.py:1148`): GPU-accelerated effects generator for MPS/CUDA
- **Helper functions** (`lucidsonicdreams/helper_functions.py`): Utility functions for audio processing, model downloading, and mathematical operations

### Audio-Visual Pipeline

1. **Audio Analysis**: Uses librosa to extract spectrograms and chromagrams from input audio
2. **Vector Generation**: Converts audio features into latent noise vectors and class vectors for StyleGAN
3. **Frame Generation**: Processes vectors through StyleGAN3 to generate image sequences
4. **Post-processing**: Applies effects and assembles final video with moviepy

### Latent Space Exploration

The fork adds latent space exploration features:
- `latent_center`: Optional parameter to define a specific point in latent space
- `latent_radius`: Controls the exploration radius around the center point
- Functions like `constrain_noise()`, `slerp()`, and `generate_narrow_perturbation()` in helper_functions.py

## Development Commands

### Installation and Setup

```bash
# Install package in development mode
pip install -e .

# Install dependencies
pip install torch librosa numpy moviepy Pillow tqdm scipy scikit-image pygit2 gdown requests pandas SoundFile
```

### Running the Project

```bash
# Basic usage - generate video from audio
python dream.py

# Apply post-processing effects to existing video
python vibify.py input_video.mp4

# Download StyleGAN models (automatic on first run)
python download_model.py
```

### Code Quality

The project uses flake8 with E501 (line length) ignored as specified in `setup.cfg`.

## Hardware Requirements

- **Apple Silicon recommended**: Optimized for MPS (Metal Performance Shaders)
- **Memory**: Significant GPU/MPS memory required for StyleGAN inference
- **Storage**: Large model files (~1GB+ for StyleGAN weights)

## Key Files and Directories

- `lucidsonicdreams/main.py`: Core pipeline implementation
- `lucidsonicdreams/helper_functions.py`: Utility functions and mathematical operations
- `lucidsonicdreams/sample_effects.py`: Pre-built visual effects (contrast, flash)
- `dream.py`: Example usage with latent space exploration
- `vibify.py`: Post-processing script for CRT-style effects
- `models/`: Directory containing StyleGAN model weights
- `stylegan3/`: Auto-cloned StyleGAN3 repository (NVLabs official)

## Architecture Notes

### StyleGAN Integration

The codebase supports both TensorFlow and PyTorch StyleGAN implementations:
- **PyTorch** (default): Used for most models, leverages MPS on Apple Silicon
- **TensorFlow**: Used for specific models like "wikiart" that can't be converted

### Memory Optimization

- Uses `concurrent.futures.ThreadPoolExecutor` for asynchronous image processing
- Implements batched inference with configurable `batch_size`
- Includes memory management for Apple Silicon (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`)
- **NEW**: True batch processing throughout pipeline for 2-4x speed improvements
- **NEW**: GPU-accelerated effects processing using PyTorch MPS
- **NEW**: Vectorized effects operations support batch arrays [B, H, W, C]

### Audio Processing Features

- **Harmonic/Percussive Separation**: Uses librosa's HPSS for different audio components
- **Multi-track Support**: Separate audio files for pulse, motion, class, contrast, and flash
- **Real-time Analysis**: Frame-by-frame audio feature extraction synchronized to video FPS

### Effects System

- **Built-in Effects**: Contrast and flash effects that react to audio
- **Custom Effects**: Plugin system for user-defined effects via `EffectsGenerator`
- **Post-processing**: Additional CRT-style effects in `vibify.py` including upscaling, chromatic aberration, and scanlines