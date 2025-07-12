# Lucid Sonic Dreams - Apple Silicon Psychedelic Layer Fork

**An Apple Silicon-optimized fork of Lucid Sonic Dreams with revolutionary neural layer visualization capabilities.**

## 🎯 What This Fork Does

This specialized fork transforms Lucid Sonic Dreams into a **psychedelic neural archaeology tool** for Apple Silicon users. Instead of generating standard StyleGAN videos, it lets you **peer inside the AI's mind** by extracting and visualizing the raw intermediate layers where abstract mathematical concepts gradually crystallize into recognizable forms.

### Key Innovations

1. **🧠 Neural Layer Extraction**: Capture any of the 15 intermediate StyleGAN3 layers (L0-L14) to see how AI "thinks"
2. **🍎 Apple Silicon Optimized**: Built specifically for Metal Performance Shaders (MPS) - **Apple Silicon only**
3. **⚡ True Early Stopping**: Massive performance gains by stopping computation at target layers
4. **🎨 Mathematical Beauty**: Transform high-dimensional neural activations into trippy RGB visuals using PCA

## 🚫 Platform Requirements

**Apple Silicon Only**: This fork is specifically designed for Apple M1/M2/M3/M4 processors with Metal Performance Shaders. CUDA support has been completely removed.

- ✅ **Supported**: MacOS with Apple Silicon (M1, M2, M3, M4)
- ❌ **Not Supported**: Intel Macs, Windows, Linux, NVIDIA GPUs

## 🧠 Layer Visualization: Seeing AI Think

Unlike the original Lucid Sonic Dreams that shows polished final outputs, this fork reveals the **raw computational process** of StyleGAN3. Each layer represents a different level of abstraction:

- **L0-L3**: Pure mathematical abstractions - flowing geometric forms, color relationships
- **L4-L8**: Emerging structures - primitive landscapes, basic shapes  
- **L9-L12**: Complex patterns - detailed but surreal environments
- **L13-L14**: Near-final quality with subtle AI "hallucinations"

## Usage

```bash
python dream.py <audio_file> [layer] [model] [options]
```

### Arguments

- `audio_file`: Input audio file (required)
- `layer`: StyleGAN layer to extract (optional, e.g., L12_276_128)
- `model`: StyleGAN model file (optional, e.g., stylegan3-r-afhqv2-512x512.pkl)

### Options

- `--latent`: Use latent vector constraints for exploration
- `--seed <number>`: Random seed for reproducible generation

### Basic Usage

```bash
# Quick start - use all defaults
python dream.py

# Normal generation (final layer)
python dream.py my_song.mp3

# 🧠 Psychedelic layer extraction (the main feature!)
python dream.py my_song.mp3 --layer L8_276_645

# Full control
python dream.py my_song.mp3 --layer L5_84_1024 --model stylegan3-r-afhqv2-512x512.pkl --seed 42
```

## 🎨 What Makes This Fork Special

### Original Lucid Sonic Dreams
- Generates polished, realistic landscapes
- Shows only the final StyleGAN output
- Works on various platforms (CUDA, CPU)

### This Apple Silicon Fork
- **Reveals the AI's "thought process"** by extracting intermediate neural layers
- **10x performance boost** for early layers through true early stopping
- **Mathematical beauty**: 512+ dimensional neural activations → trippy RGB visuals via PCA
- **Native resolution preservation**: See layers at their true computational resolution
- **Apple Silicon exclusive**: Optimized for Metal Performance Shaders

### The Neural Archaeology Experience

When you specify `--layer L5_84_1024`, you're not getting a stylized version of the final output. You're getting the **raw mathematical state** of the neural network at that exact computational step, transformed into visual form. It's like having X-ray vision into how AI creates images.

### Example Workflows

## Layer Exploration Guide

### Finding Available Layers

```bash
python explore_layers.py <model_file>
python explore_layers.py stylegan3-r-afhqv2-512x512.pkl
```

This will show all available layers like:

```
L0_36_1024    # Early: Pure abstract patterns
L5_84_1024    # Mid: Emerging structures
L10_532_287   # Late: Detailed features
L14_512_3     # Final: Clean output
```

### Layer Types

**Early Layers (L0-L5)**: Mathematical abstractions, flowing forms, pure color
**Mid Layers (L6-L10)**: Emerging structures, surreal landscapes, complex patterns
**Late Layers (L11-L14)**: Refined details, subtle artifacts, near-final quality

## Performance Features

### True Early Stopping

When using layer extraction, the system only computes up to the target layer, providing massive speed improvements:

- **L0-L5**: ~10x faster (skip most expensive layers)
- **L6-L10**: ~5x faster (skip high-resolution processing)
- **L11-L14**: Normal speed

### Native Resolution Preservation

Each layer maintains its natural resolution without forced rescaling:

- L10_532_287 outputs at 532×532 pixels
- L8_276_645 outputs at 276×276 pixels
- Use `vibify.py` for post-processing rescaling if needed

### Adaptive Batch Processing

Automatically adjusts batch size based on model resolution to prevent memory errors:

- 512×512 models: batch_size=2
- 256×256 models: batch_size=4
- Lower resolution: batch_size=8

## Example Workflows

#### 1. Pure Abstract Art (Early Layers)

```bash
# Mathematical flowing forms
python dream.py mystical_ambient.mp3 L0_36_1024

# Geometric color blobs
python dream.py electronic_beat.mp3 L2_52_1024

# Primitive shape formation
python dream.py jazz_improv.mp3 L4_84_1024
```

#### 2. Surreal Landscapes (Mid Layers)

```bash
# Dream-like natural scenes
python dream.py atmospheric_music.mp3 L8_276_645

# Complex surreal environments
python dream.py psychedelic_rock.mp3 L9_276_431

# Detailed but distorted landscapes
python dream.py ambient_drone.mp3 L10_532_287
```

#### 3. Subtle Psychedelia (Late Layers)

```bash
# Nearly normal with AI artifacts
python dream.py classical_piece.mp3 L13_512_128

# Recommended: Best balance of detail and psychedelia
python dream.py any_music.mp3 L14_512_3
```

## Post-Processing

Use `vibify.py` to add CRT-style effects to any generated video:

```bash
python vibify.py song.mp4
```

This applies upscaling, chromatic aberration, scanlines, and other retro effects.

```

#### 4. Visualization Comparison
```bash
# Create 4 different visualization methods for any layer
python better_visualization.py L8_148_512
python better_visualization.py L2_36_512
python better_visualization.py L14_256_3
```

### Programming Interface

```python
from lucidsonicdreams import LucidSonicDream

# Load the layer capture functionality
exec(open('direct_patch.py').read())

# Create instance
L = LucidSonicDream(song="music.mp3", style="models/lhq-256-stylegan3-t-25Mimg.pkl")

# Generate with layer capture
L.hallucinate(
    file_name="psychedelic_output.mp4",
    resolution=256,
    fps=24,
    capture_layer="L8_148_512"  # Specify any layer
)
```

### Complete Layer Catalog

The StyleGAN3 network contains 15 main layers (L0-L14), each producing different levels of abstraction:

#### Early Layers (Abstract Geometric Patterns)

- **L0_36_512** - First layer: Pure mathematical abstractions, flowing geometric forms
- **L1_36_512** - Basic pattern formation, color gradients
- **L2_36_512** - Blob-like forms, primary color relationships
- **L3_52_512** - Early shape recognition, basic textures
- **L4_52_512** - Primitive landscape elements, horizon hints

#### Mid Layers (Recognizable but Surreal)

- **L5_84_512** - Abstract landscapes with geological patterns
- **L6_84_512** - Natural forms emerge: rocks, clouds, water
- **L7_148_512** - More detailed terrain, atmospheric effects
- **L8_148_512** - Recognizable landscapes with dream-like distortions
- **L9_148_362** - Complex natural scenes with surreal elements

#### Late Layers (Detailed but Psychedelic)

- **L10_276_256** - Highly detailed landscapes with neural artifacts
- **L11_276_181** - Near-photorealistic with AI "hallucinations"
- **L12_276_128** - Complex scenes with subtle distortions
- **L13_256_128** - Almost final quality with minor neural effects
- **L14_256_3** - **Penultimate layer**: Final landscapes with subtle AI artifacts (recommended for psychedelic effects)

### Recommended Layers for Different Effects

- **Pure Abstraction**: L0, L1, L2 - Mathematical forms and color flows
- **Geometric Psychedelia**: L3, L4, L5 - Shape formation and primitive landscapes
- **Surreal Landscapes**: L8, L9, L10 - Recognizable but dream-like scenes
- **Subtle Psychedelia**: L13, L14 - Nearly normal with AI artifacts

### Advanced Visualization Ideas

1. **Layer Morphing**: Blend multiple layers for complex effects
2. **Temporal Layer Switching**: Change layers based on audio intensity
3. **Multi-layer Compositing**: Overlay different abstraction levels
4. **Layer-specific Color Grading**: Apply different effects to different layers

## Resources

-[NVlabs StyleGAN3](https://github.com/NVlabs/stylegan3)

- [awesome-pretrained-stylegan3](https://github.com/justinpinkney/awesome-pretrained-stylegan3)

# Fast Lucid Sonic Dreams

## Features Implemented

- Stylegan3 support;
- Improved performance through asynchronous image writing. (~1.5x on a 3090Ti - 1024x1024 output frames - 60s - 60fps)

## Features to implement

- More parameters for enhance different genres of music;
- More features for different motions.

# Lucid Sonic Dreams

Lucid Sonic Dreams syncs GAN-generated visuals to music. By default, it uses [NVLabs StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch), with pre-trained models lifted from [Justin Pinkney's consolidated repository](https://github.com/justinpinkney/awesome-pretrained-stylegan2). Custom weights and other GAN architectures can be used as well.

Sample output can be found on [YouTube](https://youtu.be/l-nGC-ve7sI) and [Instagram](https://www.instagram.com/lucidsonicdreams/).

## Installation

To install, simply run:

```pip install lucidsonicdreams```

## Usage

You may refer to the [Lucid Sonic Dreams Tutorial Notebook](https://colab.research.google.com/drive/1Y5i50xSFIuN3V4Md8TB30_GOAtts7RQD?usp=sharing) for full parameter descriptions and sample code templates. A basic visualization snippet is also found below.

### Basic Visualization

```
from lucidsonicdreams import LucidSonicDream


L = LucidSonicDream(song = 'song.mp3',
                    style = 'abstract photos')

L.hallucinate(file_name = 'song.mp4')
```

### Parameters

The Parameters

Now, the parameters can be easily understood by separating them into 7 categories: Initialization, Pulse, Motion, Class, Effects, Video, and Other.

If this is still overwhelming, it's recommended that you start off by tuning speed_fpm, pulse_react, motion_react and class_pitch_react, and build from there. These parameters make the biggest difference.
Initialization

    speed_fpm (Default: 12) - FPM stands for "Frames per Minute". This determines how many images are initialized - the more there are, the faster the visuals morph. If speed_fpm = 0, then only one image is initialized, and that single image reacts to the audio. In this case, there will be no motion during silent parts of the audio.

Pulse Parameters

    pulse_react (Default: 0.5) - The "strength" of the pulse. It is recommended to keep this between 0 and 2.

    pulse_percussive (Default: True) - If True while pulse_harmonic is False, pulse reacts to the audio's percussive elements.

    pulse_harmonic (Default: False) - If True while pulse_percussive is False, pulse reacts to the audio's harmonic elements.

    Note: If both parameters are True or both parameters are False, pulse reacts to the "entire" unaltered audio.

    pulse_audio - Path to a separate audio file to be used to control pulse. This is recommended if you have access to an isolated drum/percussion track. If passed, pulse_percussive and pulse_harmonic are ignored. Note: this parameter is passed when defining the LucidSonicDream object.

Motion Parameters

    motion_react (0.5), motion_percussive (False), motion_harmonic (True), and motion_audio - Simply the "motion" equivalents of the pulse parameters above.
    motion_randomness (Default: 0.5)- Degree of randomness of motion. Higher values will typically prevent the video from cycling through the same visuals repeatedly. Must range from 0 to 1.
    truncation (Default: 1) - Controls the variety of visuals generated. Lower values lead to lower variety. Note: A very low value will usually lead to "jittery" visuals. Must range from 0 to 1.

Class Parameters

(Note: Most of these parameters were heavily inspired by the Deep Music Visualizer project by Matt Siegelman)

    classes - List of at most 12 numerical object labels. If none, 12 labels are selected at random.
    dominant_classes_first (Default: False)- If True, the list passed to "classes" is sorted by prominence in descending order.
    class_pitch_react (Default: 0.5)- Class equivalent of pulse_react and motion_react. It is recommended to keep this between 0 and 2.
    class_smooth_seconds (Default: 1) - Number of seconds spent smoothly interpolating between each class vector. The higher the value, the less "sudden" the change of class.
    class_complexity (Default: 1) - Controls the "complexity" of images generated. Lower values tend to generate more simple and mundane images, while higher values tend to generate more intricate and bizzare objects. It is recommended to keep this between 0 and 1.
    class_shuffle_seconds (Default: None) - Controls the timestamps wherein the mapping of label to note is re-shuffled. This is recommended when the audio used has a limited range of pitches, but you wish for more classes to be shown. If the value passed is a number n, classes are shuffled every n seconds. If the value passed is a list of numbers, these numbers are used as timestamps (in seconds) wherein classes are shuffled.
    class_shuffle_strength (Default: 0.5) - Controls how drastically classes are re-shuffled. Only applies when class_shuffle_seconds is passed. It is recommended to keep this between 0 and 1.
    class_audio - Class equivalent of pulse_audio and motion_audio. Passed when defining the LucidSonicDream object.

Effects Parameters

    contrast_strength (Default: 0.5) - Strength of default contrast effect. It is recommended to keep this between 0 and 1.

    contrast_percussive (Default: True) - If true, contrast reacts to the audio's percussive elements. Must range from 0 to 1.

    contrast_audio - Equivalent of previous "audio" arguments. Passed when defining the LucidSonicDream object.

    Note: If none of these arguments are passed, the contrast effect will not be applied.

    flash_strength (0.5), flash_percussive (True), and flash_audio - Equivalent of the previous three parameters, but for the a "flash" effect. It is recommended to keep these between 0 and 1. If none of these arguments are passed, the flash effect will not be applied.

    custom_effects - List of custom, user-defined effects to apply (See B.4)

Video Parameters

    resolution - Self-explanatory. Low resolutions are recommended for "trial" renders. If none is passed, unaltered high-resolution images will be used.
    start (Default: 0) - Starting timestamp in seconds.
    duration - Video duration in seconds. If none is passed, full duration of audio will be used.
    output_audio - Final output audio of the video. Overwrites audio from "song" parameter if provided (See B.5)
    fps (Default: 43) - Video Frames Per Second.
    save_frames (Default: False) - If true, saved all individual video frames on disk.

Other

    batch_size (Default: 1) - Determines how many vectors are simoultaneously fed to the model. Larger batch sizes are much faster, but cost more GPU memory
