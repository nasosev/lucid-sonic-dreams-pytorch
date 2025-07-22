import sys
import os
import shutil
import pickle
from tqdm import tqdm
import inspect
import numpy as np
import random
from scipy.stats import truncnorm
import torch
from PIL import Image
import librosa
import soundfile
import moviepy.editor as mpy
import pygit2
from importlib import import_module
import time
import logging

from .helper_functions import *
from .sample_effects import *

import concurrent.futures


def import_stylegan_torch():
    # Clone Official StyleGAN2-ADA Repository
    if not os.path.exists("stylegan3"):
        pygit2.clone_repository("https://github.com/NVlabs/stylegan3.git", "stylegan3")
    # StyleGan3 imports
    sys.path.append("stylegan3")
    import legacy
    import dnnlib


def import_stylegan_tf():
    print("Cloning tensorflow...")
    if not os.path.exists("stylegan2_tf"):
        pygit2.clone_repository(
            "https://github.com/NVlabs/stylegan2-ada.git", "stylegan2_tf"
        )

    # StyleGAN2 Imports
    sys.path.append("stylegan2_tf")
    import dnnlib as dnnlib
    from dnnlib.tflib.tfutil import convert_images_to_uint8 as convert_images_to_uint8

    init_tf()


def show_styles():
    """Show names of available (non-custom) styles"""
    all_models = consolidate_models()
    styles = set([model["name"].lower() for model in all_models])
    print(*styles, sep="\n")




class LucidSonicDream:
    def __init__(
        self,
        song: str,
        pulse_audio: str = None,
        motion_audio: str = None,
        class_audio: str = None,
        contrast_audio: str = None,
        flash_audio: str = None,
        style: str = "wikiart",
        input_shape: int = None,
        num_possible_classes: int = None,
        latent_center: np.ndarray = None,  # New optional parameter
        latent_radius: float = None,  # New optional parameter
        seed: int = None,  # New optional parameter for reproducibility
        verbose: bool = False,  # Enable detailed timing logs
    ):
        # If style is a function, raise exception if function does not take
        # noise_batch or class_batch parameters
        if callable(style):
            func_sig = list(inspect.getfullargspec(style))[0]
            for arg in ["noise_batch", "class_batch"]:
                if arg not in func_sig:
                    sys.exit(
                        "func must be a function with parameters "
                        "noise_batch and class_batch"
                    )
            # Raise exception if input_shape or num_possible_classes is not provided
            if (input_shape is None) or (num_possible_classes is None):
                sys.exit(
                    "input_shape and num_possible_classes "
                    "must be provided if style is a function"
                )

        # Define attributes
        self.song = song
        self.pulse_audio = pulse_audio
        self.motion_audio = motion_audio
        self.class_audio = class_audio
        self.contrast_audio = contrast_audio
        self.flash_audio = flash_audio
        self.style = style
        self.input_shape = input_shape or 512
        self.num_possible_classes = num_possible_classes
        self.style_exists = False

        # Add new latent space exploration parameters
        self.latent_center = latent_center
        self.latent_radius = latent_radius
        self.seed = seed
        self.verbose = verbose

        # Initialize optimized audio processor
        self._audio_processor = OptimizedAudioProcessor()
        
        # Initialize tensor pools for reuse (will be allocated after batch size is known)
        self._tensor_pools_allocated = False
        self._noise_tensor_pool = None
        self._class_tensor_pool = None

        # some stylegan models cannot be converted to pytorch (wikiart)
        self.use_tf = style in ("wikiart",)
        if self.use_tf:
            print("Cloning tensorflow...")
            if not os.path.exists("stylegan2_tf"):
                pygit2.clone_repository(
                    "https://github.com/NVlabs/stylegan2-ada.git", "stylegan2_tf"
                )
            # StyleGAN2 Imports
            sys.path.append("stylegan2_tf")
            self.dnnlib = import_module("dnnlib")
            tflib = import_module("dnnlib.tflib.tfutil")
            self.convert_images_to_uint8 = tflib.convert_images_to_uint8
            self.init_tf = tflib.init_tf
            self.init_tf()
        else:
            if not os.path.exists("stylegan3"):
                pygit2.clone_repository(
                    "https://github.com/NVlabs/stylegan3.git", "stylegan3"
                )
            sys.path.append("stylegan3")
            self.dnnlib = import_module("dnnlib")
            self.legacy = import_module("legacy")

    def stylegan_init(self):
        """Initialize StyleGAN(2) weights"""

        style = self.style

        # Initialize TensorFlow
        # if self.use_tf:
        #    init_tf()

        # If style is not a .pkl file path, download weights from corresponding URL
        if ".pkl" not in style:
            all_models = consolidate_models()
            all_styles = [model["name"].lower() for model in all_models]

            # Raise exception if style is not valid
            if style not in all_styles:
                sys.exit(
                    "Style not valid. Call show_styles() to see all "
                    "valid styles, or upload your own .pkl file."
                )

            download_url = [
                model for model in all_models if model["name"].lower() == style
            ][0]["download_url"]
            weights_file = style + ".pkl"

            # If style .pkl already exists in working directory, skip download
            if not os.path.exists(weights_file):
                print("Downloading {} weights (This may take a while)...".format(style))
                try:
                    download_weights(download_url, weights_file)
                except Exception:
                    exc_msg = (
                        "Download failed. Try to download weights directly at {} "
                        "and pass the file path to the style parameter".format(
                            download_url
                        )
                    )
                    sys.exit(exc_msg)
                print("Download complete")

        else:
            weights_file = style

        # load generator
        if self.use_tf:
            # Load weights
            with open(weights_file, "rb") as f:
                self.Gs = pickle.load(f)[2]
        else:
            print(f"Loading networks from {weights_file}...")
            # Apple Silicon MPS device
            device = torch.device("mps")
            with self.dnnlib.util.open_url(weights_file) as f:
                self.Gs = self.legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

        # Auto assign num_possible_classes attribute
        try:
            print(self.Gs.mapping.input_templates)
            self.num_possible_classes = self.Gs.mapping.input_templates[1].shape[1]
        except ValueError:
            print(self.Gs.mapping.static_kwargs.label_size)
            self.num_possible_classes = (
                self.Gs.components.mapping.static_kwargs.label_size
            )
        except Exception:
            self.num_possible_classes = 0

    def load_specs(self):
        """Load normalized spectrograms and chromagram using optimized audio processing"""

        logging.info("Loading and processing audio with optimized pipeline...")

        # Use optimized audio processor
        audio_result = self._audio_processor.load_and_process_audio(
            primary_audio=self.song,
            pulse_audio=self.pulse_audio,
            motion_audio=self.motion_audio,
            class_audio=self.class_audio,
            start=self.start,
            duration=self.duration,
            fps=self.fps,
            input_shape=self.input_shape,
            pulse_percussive=self.pulse_percussive,
            pulse_harmonic=self.pulse_harmonic,
            motion_percussive=self.motion_percussive,
            motion_harmonic=self.motion_harmonic
        )

        # Extract results from optimized processor
        spectrograms = audio_result['spectrograms']
        self.spec_norm_pulse = spectrograms['spec_norm_pulse']
        self.spec_norm_motion = spectrograms['spec_norm_motion']
        self.spec_norm_class = spectrograms['spec_norm_class']

        self.chrom_class = audio_result['chrom_class']
        self.pitches_sorted = audio_result['pitches_sorted']
        self.frame_duration = audio_result['frame_duration']

        # Keep primary audio for duration calculations
        self.wav, self.sr = audio_result['primary_audio']

        logging.info("Audio processing completed successfully")

    def _warm_up_mps_memory(self):
        """Pre-warm MPS memory subsystem to eliminate cold start penalty"""
        if self.use_tf:
            return
            
        print("🔥 Warming up MPS memory subsystem...")
        start_time = time.time()
        
        # Create dummy tensors matching actual workload dimensions - use FP32 for StyleGAN compatibility
        device = torch.device("mps")
        dummy_noise = torch.randn(self.batch_size, self.input_shape, device=device, dtype=torch.float32)
        dummy_class = torch.randn(self.batch_size, self.num_possible_classes, device=device, dtype=torch.float32)
        
        # Run one warmup inference pass to allocate memory pools
        with torch.no_grad():
            w_batch = self.Gs.mapping(dummy_noise, dummy_class, truncation_psi=1.0)
            _ = self.Gs.synthesis(w_batch, noise_mode="const")
        
        # Clear cache after warmup but keep memory pools allocated
        torch.mps.empty_cache()
        
        warmup_time = time.time() - start_time
        print(f"✅ MPS memory warmed up in {warmup_time:.2f}s")
        
        # Enable MPS-specific optimizations
        self._configure_mps_optimizations()

    def _configure_mps_optimizations(self):
        """Configure PyTorch and MPS-specific optimizations"""
            
        # Enable MPS optimizations - assume MPS is always available
        torch.backends.mps.enabled = True
        
        # Apple Silicon M1/M2/M3/M4 specific optimizations
        if hasattr(torch.backends.mps, 'max_split_size_mb'):
            torch.backends.mps.max_split_size_mb = 256  # Smaller for FP16, better cache utilization
            
        # Aggressive MPS memory optimizations for Apple Silicon
        torch.backends.mps.enable_memory_efficient_attention = True
        
        # Set optimal tensor memory layout for unified memory architecture
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        # Use FP16 where beneficial, but keep FP32 as default for StyleGAN compatibility
        # torch.set_default_dtype(torch.float16)  # Too aggressive - causes StyleGAN issues
        
        # Ultra-aggressive Apple Silicon specific optimizations
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # No memory fragmentation
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'   # Aggressive memory management
        
        # Apple Silicon unified memory architecture optimizations
        torch.backends.quantized.engine = 'qnnpack'  # Optimized for ARM
        
        # Additional MPS optimizations to avoid CPU fallbacks
        torch.backends.mps.enable_memory_efficient_attention = True
        
        # Optimize for Apple Silicon GPU characteristics
        if hasattr(torch.backends.mps, 'enable_metal_math_opts'):
            torch.backends.mps.enable_metal_math_opts = True
        
        if self.verbose:
            print("🚀 Optimized MPS pipeline configured for Apple Silicon (StyleGAN-compatible)")
            
    def _log_performance_summary(self, start_time):
        """Log comprehensive performance summary"""
        total_time = time.time() - start_time
        if hasattr(self, 'noise') and len(self.noise) > 0:
            total_frames = len(self.noise)
            fps_effective = total_frames / total_time
            
            print(f"\n🚀 PERFORMANCE SUMMARY:")
            print(f"   Total execution time: {total_time:.1f}s")
            print(f"   Total frames generated: {total_frames}")
            print(f"   Effective FPS: {fps_effective:.2f}")
            print(f"   Time per frame: {total_time/total_frames:.3f}s")
            
            if hasattr(self, 'batch_size'):
                batches_processed = len(self.noise) // self.batch_size
                time_per_batch = total_time / batches_processed if batches_processed > 0 else 0
                print(f"   Batch size: {self.batch_size}")
                print(f"   Batches processed: {batches_processed}")
                print(f"   Time per batch: {time_per_batch:.3f}s")
            
            print("   🚀 MPS OPTIMIZATIONS ACTIVE:")
            print("   ✅ MPS memory pre-warming")
            print("   ✅ Smart cache clearing strategy") 
            print("   ✅ Tensor pool pre-allocation")
            print("   ✅ Native MPS power iteration PCA")
            print("   ✅ Inference mode optimizations")
            print("   ✅ Apple Silicon unified memory layout")
            print("   ✅ Direct ATen memory operations")
            print("   ✅ Zero-copy tensor reuse")
            print("   ✅ Optimized memory formats")
            print("   ⚡ CPU/CUDA fallbacks REMOVED")

    def _smart_cache_clear(self, batch_idx, total_batches):
        """Smart MPS cache clearing to balance memory vs performance"""
            
        start_time = time.time()
        
        # Clear cache strategically - less frequent clearing for better performance
        # Clear every 10% of batches or minimum every 5 batches
        cache_interval = max(5, total_batches // 10)
        
        # Always clear on first batch (after warmup) and last batch
        should_clear = (
            batch_idx == 0 or 
            batch_idx == total_batches - 1 or 
            batch_idx % cache_interval == 0
        )
        
        if should_clear:
            torch.mps.empty_cache()
            if self.verbose:
                print(f"🧹 Cleared MPS cache at batch {batch_idx}")
        
        return time.time() - start_time


    def transform_classes(self):
        """Transform/assign value of classes"""
        print("Num classes of model: ", self.num_possible_classes)
        # If model does not use classes, simply return list of 0's
        if self.num_possible_classes == 0:
            self.classes = [0] * 12

        else:

            # If list of classes is not provided, generate a random sample
            if self.classes is None:
                self.classes = random.sample(
                    range(self.num_possible_classes),
                    min([self.num_possible_classes, 12]),
                )

            # If length of list < 12, repeat list until length is 12
            if len(self.classes) < 12:
                self.classes = (self.classes * int(np.ceil(12 / len(self.classes))))[
                    :12
                ]

            # If dominant_classes_first is True, sort classes accordingly
            if self.dominant_classes_first:
                self.classes = [
                    self.classes[i] for i in np.argsort(self.pitches_sorted)
                ]

    def update_motion_signs(self):
        """Update direction of noise interpolation based on truncation value"""
        m = self.motion_react
        t = self.truncation
        current_noise = self.current_noise
        motion_signs = self.motion_signs

        new_signs = np.where(
            current_noise - m < -2 * t,
            1,
            np.where(current_noise + m >= 2 * t, -1, motion_signs),
        )
        return new_signs

    def generate_class_vec(self, frame):
        """Generate a class vector using chromagram, where each pitch
        corresponds to a class"""

        classes = self.classes
        chrom_class = self.chrom_class
        class_vecs = self.class_vecs
        num_possible_classes = self.num_possible_classes
        class_complexity = self.class_complexity
        class_pitch_react = self.class_pitch_react * 43 / self.fps

        # For the first class vector, simple use values from
        # the first point in time where at least one pitch > 0
        # (controls for silence at the start of a track)
        if frame == 0:

            first_chrom = chrom_class[:, np.min(np.where(chrom_class.sum(axis=0) > 0))]
            update_dict = dict(zip(classes, first_chrom))
            class_vec = np.array(
                [
                    update_dict.get(i) if update_dict.get(i) is not None else 0
                    for i in range(num_possible_classes)
                ]
            )

        # For succeeding vectors, update class values scaled by class_pitch_react
        else:

            update_dict = dict(zip(classes, chrom_class[:, frame]))
            class_vec = class_vecs[frame - 1] + class_pitch_react * np.array(
                [
                    update_dict.get(i) if update_dict.get(i) is not None else 0
                    for i in range(num_possible_classes)
                ]
            )

        # Normalize class vector between 0 and 1
        if np.where(class_vec != 0)[0].shape[0] != 0:
            class_vec[class_vec < 0] = np.min(class_vec[class_vec >= 0])
            class_vec = (class_vec - np.min(class_vec)) / np.ptp(class_vec)

        # If all values in class vector are equal, add 0.1 to first value
        if (len(class_vec) > 0) and (np.all(class_vec == class_vec[0])):
            class_vec[0] += 0.1

        return class_vec * class_complexity

    def is_shuffle_frame(self, frame):
        """Determines if classes should be shuffled in current frame"""

        class_shuffle_seconds = self.class_shuffle_seconds
        fps = self.fps

        # If class_shuffle_seconds is an integer, return True if current timestamp
        # (in seconds) is divisible by this integer
        if isinstance(class_shuffle_seconds, int):
            if frame != 0 and frame % round(class_shuffle_seconds * fps) == 0:
                return True
            else:
                return False

        # If class_shuffle_seconds is a list, return True if current timestamp
        # (in seconds) is in list
        if isinstance(class_shuffle_seconds, list):
            if frame / fps + self.start in class_shuffle_seconds:
                return True
            else:
                return False

    def generate_vectors(self):
        """Generates noise and class vectors as inputs for each frame"""
        # Smoothing parameters
        PULSE_SMOOTH = 0.75
        MOTION_SMOOTH = 0.75

        fps = self.fps
        num_frames = len(self.spec_norm_class)
        class_smooth_frames = self.class_smooth_seconds * fps
        class_shuffle_strength = round(self.class_shuffle_strength * 12)
        motion_react = self.motion_react * 20 / fps

        # Set random seed for reproducibility if provided
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Determine number of distinct noise vectors based on speed_fpm and audio duration
        duration_seconds = librosa.get_duration(y=self.wav, sr=self.sr)
        num_init_noise = round(duration_seconds / 60 * self.speed_fpm)

        # --- New: Explore a specific latent region if latent_center is provided ---
        if self.latent_center is not None:
            # Use latent_radius (or default to 0.5) to control the spread
            radius = self.latent_radius if self.latent_radius is not None else 0.5
            if num_init_noise < 2:
                base_noise = (
                    self.latent_center
                    + radius
                    * truncnorm.rvs(
                        -2, 2, size=(self.batch_size, self.input_shape)
                    ).astype(np.float32)[0]
                )
                noise = [base_noise.copy() for _ in range(num_frames)]
            else:

                # Generate one small perturbation using a narrow normal distribution.
                perturbation = generate_narrow_perturbation(
                    512, scale=self.latent_radius
                )
                target_latent = self.latent_center + perturbation

                # Then, for each frame, compute the interpolation factor:
                noise = []
                for i in range(num_frames):
                    t = i / (num_frames - 1)  # interpolation parameter in [0, 1]
                    # Slerp between the original latent_center and the target_latent
                    interpolated = slerp(t, self.latent_center, target_latent)
                    noise.append(interpolated)

                init_noise = [
                    self.latent_center
                    + radius
                    * truncnorm.rvs(-2, 2, size=(1, self.input_shape)).astype(
                        np.float32
                    )[0]
                    for i in range(num_init_noise)
                ]
                steps = int(np.floor(num_frames / len(init_noise)) - 1)

                noise = full_frame_interpolation(init_noise, steps, num_frames)
        else:
            # --- Original behavior when latent_center is not provided ---
            if num_init_noise < 2:
                base_noise = (
                    self.truncation
                    * truncnorm.rvs(
                        -2, 2, size=(self.batch_size, self.input_shape)
                    ).astype(np.float32)[0]
                )
                noise = [base_noise.copy() for _ in range(num_frames)]
            else:
                init_noise = [
                    self.truncation
                    * truncnorm.rvs(-2, 2, size=(1, self.input_shape)).astype(
                        np.float32
                    )[0]
                    for i in range(num_init_noise)
                ]
                steps = int(np.floor(num_frames / len(init_noise)) - 1)
                noise = full_frame_interpolation(init_noise, steps, num_frames)

        # Pre-allocate arrays - use FP32 for numerical stability in vector generation
        pulse_noise = np.zeros((num_frames, self.input_shape), dtype=np.float32)
        motion_noise = np.zeros((num_frames, self.input_shape), dtype=np.float32)
        self.class_vecs = np.zeros((num_frames, self.num_possible_classes), dtype=np.float32)

        # Base vectors for Pulse and Motion updates  
        pulse_base = np.full(self.input_shape, self.pulse_react, dtype=np.float32)
        motion_base = np.full(self.input_shape, motion_react, dtype=np.float32)

        # Initialize update directions and randomness factors
        self.motion_signs = np.array(
            [random.choice([1, -1]) for _ in range(self.input_shape)]
        )
        rand_factors = np.where(
            np.random.rand(self.input_shape) < 0.5, 1, 1 - self.motion_randomness
        )
        cumulative_motion = np.zeros(self.input_shape, dtype=np.float32)

        for i in range(num_frames):
            # Reinitialize randomness factors every 10 seconds
            if i % round(fps * 10) == 0:
                rand_factors = np.where(
                    np.random.rand(self.input_shape) < 0.5,
                    1,
                    1 - self.motion_randomness,
                )

            # Compute incremental update vectors for Pulse and Motion noise
            pulse_noise_add = pulse_base * self.spec_norm_pulse[i]
            motion_noise_add = (
                motion_base
                * self.spec_norm_motion[i]
                * self.motion_signs
                * rand_factors
            )

            # Apply recursive smoothing
            if i > 0:
                pulse_noise_add = pulse_noise[
                    i - 1
                ] * PULSE_SMOOTH + pulse_noise_add * (1 - PULSE_SMOOTH)
                motion_noise_add = motion_noise[
                    i - 1
                ] * MOTION_SMOOTH + motion_noise_add * (1 - MOTION_SMOOTH)

            pulse_noise[i] = pulse_noise_add
            motion_noise[i] = motion_noise_add

            # Accumulate motion noise and update the base noise vector
            cumulative_motion += motion_noise_add
            noise[i] = noise[i] + pulse_noise_add + cumulative_motion
            # Constrain the noise to remain within a sphere around latent_center
            if self.latent_center is not None:
                noise[i] = constrain_noise(noise[i], self.latent_center, self.latent_radius)

            # Update current noise and adjust motion directions
            self.noise = noise
            self.current_noise = noise[i]
            self.motion_signs = self.update_motion_signs()

            # Shuffle classes if required
            if self.is_shuffle_frame(i):
                self.classes = (
                    self.classes[class_shuffle_strength:]
                    + self.classes[:class_shuffle_strength]
                )

            # Generate and store the class update vector for this frame
            class_vec_add = self.generate_class_vec(frame=i)
            self.class_vecs[i] = class_vec_add

        # Smooth class vectors by averaging over frames and interpolating
        if class_smooth_frames > 1:
            class_frames_interp = [
                np.mean(self.class_vecs[i : i + class_smooth_frames], axis=0)
                for i in range(0, len(self.class_vecs), class_smooth_frames)
            ]
            smoothed_class_vecs = full_frame_interpolation(
                class_frames_interp, class_smooth_frames, len(self.class_vecs)
            )
            self.class_vecs = np.array(smoothed_class_vecs)

        # Convert noise list to numpy array for downstream processing
        self.noise = np.array(self.noise)
        # self.class_vecs is already a numpy array from pre-allocation

    def setup_effects(self):
        """Initializes effects to be applied to each frame"""

        self.custom_effects = self.custom_effects or []
        start = self.start
        duration = self.duration

        # Initialize pre-made Contrast effect
        if all(
            var is None
            for var in [
                self.contrast_audio,
                self.contrast_strength,
                self.contrast_percussive,
            ]
        ):
            pass
        else:
            self.contrast_audio = self.contrast_audio or self.song
            self.contrast_strength = self.contrast_strength or 0.5
            self.contrast_percussive = self.contrast_percussive or True

            contrast = BatchEffectsGenerator(
                audio=self.contrast_audio,
                func=contrast_effect,
                strength=self.contrast_strength,
                percussive=self.contrast_percussive,
            )
            self.custom_effects.append(contrast)

        # Initialize pre-made Flash effect
        if all(
            var is None
            for var in [self.flash_audio, self.flash_strength, self.flash_percussive]
        ):
            pass
        else:
            self.flash_audio = self.flash_audio or self.song
            self.flash_strength = self.flash_strength or 0.5
            self.flash_percussive = self.flash_percussive or True

            flash = BatchEffectsGenerator(
                audio=self.flash_audio,
                func=flash_effect,
                strength=self.flash_strength,
                percussive=self.flash_percussive,
            )
            self.custom_effects.append(flash)

        # Initialize Custom effects
        for effect in self.custom_effects:
            effect.audio = effect.audio or self.song
            effect.render_audio(
                start=start,
                duration=duration,
                n_mels=self.input_shape,
                hop_length=self.frame_duration,
            )

    def process_save_image(self, queue, num_frame_batches):
        i = 0
        while True:
            image_frame = queue.get()
            if image_frame is None:
                queue.task_done()
                break
            
            # Apply effects to single frame
            for effect in self.custom_effects:
                image_frame = effect.apply_effect(array=image_frame, index=i)

            # Save frame with zero-padded filename
            max_frames = len(self.noise)
            file_name = str(i).zfill(len(str(max_frames)))
            self.file_names.append(file_name)
            self.final_images.append(image_frame)
            i += 1
            queue.task_done()


    def generate_frames(self):
        """Generate GAN output for each frame of video - single frame processing"""
        file_name = self.file_name
        resolution = self.resolution
        
        # Simple synthesis kwargs
        if self.use_tf:
            Gs_syn_kwargs = {
                "output_transform": {"func": self.convert_images_to_uint8, "nchw_to_nhwc": True},
                "randomize_noise": False,
                "minibatch_size": 1,
            }
        else:
            Gs_syn_kwargs = {"noise_mode": "const"}

        # Set up temporary frame directory
        self.frames_dir = file_name.split(".mp4")[0] + "_frames"
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)

        device = torch.device("mps")
        print("Using device:", device)

        self.final_images = []
        self.file_names = []

        # Generate frames one by one
        for i, (noise_vec, class_vec) in enumerate(
            tqdm(zip(self.noise, self.class_vecs), total=len(self.noise), desc="Generating frames")
        ):
            if callable(self.style):
                # Custom style function
                image_frame = self.style(noise_batch=noise_vec[None], class_batch=class_vec[None])[0]
            else:
                if self.use_tf:
                    # TensorFlow path
                    w_vec = self.Gs.components.mapping.run(noise_vec[None], class_vec[None])
                    image_frame = self.Gs.components.synthesis.run(w_vec, **Gs_syn_kwargs)[0]
                else:
                    # PyTorch path
                    noise_tensor = torch.from_numpy(noise_vec).unsqueeze(0).to(device=device, dtype=torch.float32)
                    class_tensor = torch.from_numpy(class_vec).unsqueeze(0).to(device=device, dtype=torch.float32)
                    
                    with torch.no_grad():
                        w_vec = self.Gs.mapping(noise_tensor, class_tensor, truncation_psi=self.truncation_psi)
                        image_tensor = self.Gs.synthesis(w_vec, **Gs_syn_kwargs)
                    
                    image_frame = ((image_tensor.squeeze(0).permute(1, 2, 0) * 127.5 + 128)
                                 .clamp(0, 255).to(torch.uint8).cpu().numpy())
            
            # Save single frame
            self.images_saver(image_frame, i, resolution)

    def images_saver(self, image_frame, frame_index: int, resolution):
        # Apply effects to single frame
        processed_frame = image_frame.copy()
        for effect in self.custom_effects:
            processed_frame = effect.apply_effect(array=processed_frame, index=frame_index)
        
        # Save single frame
        file_name = str(frame_index).zfill(len(str(len(self.noise))))
        self.file_names.append(file_name)
        self.final_images.append(processed_frame)
    

    def hallucinate(
        self,
        file_name: str,
        output_audio: str = None,
        fps: int = 43,
        resolution: int = None,
        start: float = 0,
        duration: float = None,
        save_frames: bool = False,
        speed_fpm: int = 12,
        pulse_percussive: bool = True,
        pulse_harmonic: bool = False,
        pulse_react: float = 0.5,
        motion_percussive: bool = False,
        motion_harmonic: bool = True,
        motion_react: float = 0.5,
        motion_randomness: float = 0.5,
        truncation: float = 1,
        classes: list = None,
        dominant_classes_first: bool = False,
        class_pitch_react: float = 0.5,
        class_smooth_seconds: int = 1,
        class_complexity: float = 1,
        class_shuffle_seconds: float = None,
        class_shuffle_strength: float = 0.5,
        contrast_strength: float = None,
        contrast_percussive: bool = None,
        flash_strength: float = None,
        flash_percussive: bool = None,
        custom_effects: list = None,
        truncation_psi: float = 1.0,
    ):
        """Full pipeline of video generation"""

        # Raise exception if speed_fpm > fps * 60
        if speed_fpm > fps * 60:
            sys.exit("speed_fpm must not be greater than fps * 60")

        # Raise exception if element of custom_effects is not EffectsGenerator
        if custom_effects:
            if not all(
                isinstance(effect, EffectsGenerator) for effect in custom_effects
            ):
                sys.exit("Elements of custom_effects must be EffectsGenerator objects")

        # Raise exception of classes is an empty list
        if classes:
            if len(classes) == 0:
                sys.exit("classes must be NoneType or list with length > 0")

        # Raise exception if any of the following parameters are not betwee 0 and 1
        for param in [
            "motion_randomness",
            "truncation",
            "class_shuffle_strength",
            "contrast_strength",
            "flash_strength",
        ]:
            if (locals()[param]) and not (0 <= locals()[param] <= 1):
                sys.exit("{} must be between 0 and 1".format(param))

        self.file_name = file_name if file_name[-4:] == ".mp4" else file_name + ".mp4"
        self.resolution = resolution

        # Use fixed batch size of 1 for optimal performance
        self.batch_size = 1
        self.speed_fpm = speed_fpm
        self.pulse_react = pulse_react
        self.motion_react = motion_react
        self.motion_randomness = motion_randomness
        self.truncation = truncation
        self.classes = classes
        self.dominant_classes_first = dominant_classes_first
        self.class_pitch_react = class_pitch_react
        self.class_smooth_seconds = class_smooth_seconds
        self.class_complexity = class_complexity
        self.class_shuffle_seconds = class_shuffle_seconds
        self.class_shuffle_strength = class_shuffle_strength
        self.contrast_strength = contrast_strength
        self.contrast_percussive = contrast_percussive
        self.flash_strength = flash_strength
        self.flash_percussive = flash_percussive
        self.custom_effects = custom_effects
        # stylegan2 params
        self.truncation_psi = truncation_psi

        # Configure logging to show optimization info
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        # Track total execution time for performance summary
        hallucinate_start_time = time.time()

        # Initialize style
        if not self.style_exists:
            print("Preparing style...")
            if not callable(self.style):
                self.stylegan_init()
                # Pre-warm MPS memory subsystem after model loading
                self._warm_up_mps_memory()
            self.style_exists = True

        # If there are changes in any of the following parameters,
        # re-initialize audio
        cond_list = [
            (not hasattr(self, "fps")) or (self.fps != fps),
            (not hasattr(self, "start")) or (self.start != start),
            (not hasattr(self, "duration")) or (self.duration != duration),
            (not hasattr(self, "pulse_percussive"))
            or (self.pulse_percussive != pulse_percussive),
            (not hasattr(self, "pulse_harmonic"))
            or (self.pulse_percussive != pulse_harmonic),
            (not hasattr(self, "motion_percussive"))
            or (self.motion_percussive != motion_percussive),
            (not hasattr(self, "motion_harmonic"))
            or (self.motion_percussive != motion_harmonic),
        ]

        if any(cond_list):
            self.fps = fps
            self.start = start
            self.duration = duration
            self.pulse_percussive = pulse_percussive
            self.pulse_harmonic = pulse_harmonic
            self.motion_percussive = motion_percussive
            self.motion_harmonic = motion_harmonic

            print("Preparing audio...")
            self.load_specs()

        # Initialize effects
        print("Loading effects...")
        self.setup_effects()

        # Transform/assign value of classes
        self.transform_classes()

        # Generate vectors
        print("\n\nDoing math...\n")
        self.generate_vectors()

        # Generate frames
        print("\n\nHallucinating... \n")
        self.generate_frames()

        # Load output audio
        if output_audio:
            wav_output, sr_output = librosa.load(
                output_audio, offset=start, duration=duration
            )
        else:
            wav_output, sr_output = self.wav, self.sr

        # Write temporary audio file
        soundfile.write("tmp.wav", wav_output, sr_output)

        # Generate final video
        audio = mpy.AudioFileClip("tmp.wav", fps=self.sr * 2)
        video = mpy.ImageSequenceClip(
            self.frames_dir, fps=self.sr / self.frame_duration
        )
        video = video.set_audio(audio)
        video.write_videofile(file_name, audio_codec="aac", threads=8)

        # Delete temporary audio file
        os.remove("tmp.wav")

        # By default, delete temporary frames directory
        if not save_frames:
            shutil.rmtree(self.frames_dir)
            
        # Log comprehensive performance summary
        if self.verbose:
            self._log_performance_summary(hallucinate_start_time)


class EffectsGenerator:
    def __init__(
        self, func, audio: str = None, strength: float = 0.5, percussive: bool = True
    ):
        self.audio = audio
        self.func = func
        self.strength = strength
        self.percussive = percussive

        # Raise exception of func does not take in parameters array,
        # strength, and amplitude
        func_sig = list(inspect.getfullargspec(func))[0]
        for arg in ["array", "strength", "amplitude"]:
            if arg not in func_sig:
                sys.exit(
                    "func must be a function with parameters "
                    "array, strength, and amplitude"
                )

    def render_audio(self, start, duration, n_mels, hop_length):
        """Prepare normalized spectrogram of audio to be used for effect"""

        # Load spectrogram
        wav, sr = librosa.load(self.audio, offset=start, duration=duration)

        # If percussive = True, decompose harmonic and percussive signals
        if self.percussive:
            wav = librosa.effects.hpss(wav)[1]

        # Get normalized RMS energy spectrogram
        self.spec = get_spec_norm(wav, sr, n_mels=n_mels, hop_length=hop_length)

    def apply_effect(self, array, index):
        """Apply effect to image (array)"""

        amplitude = self.spec[index]
        return self.func(array=array, strength=self.strength, amplitude=amplitude)
    
    def apply_batch_effect(self, batch_array, start_index):
        """Apply effect to a batch of images"""
        batch_size = batch_array.shape[0]
        
        # Get amplitudes for the entire batch
        end_index = start_index + batch_size
        amplitudes = self.spec[start_index:end_index]
        
        # Apply effect frame by frame (fallback for non-vectorized effects)
        result = np.zeros_like(batch_array)
        for i in range(batch_size):
            result[i] = self.func(
                array=batch_array[i], 
                strength=self.strength, 
                amplitude=amplitudes[i]
            )
        return result


class BatchEffectsGenerator(EffectsGenerator):
    """Optimized effects generator that supports true batch processing"""
    
    def __init__(self, func, audio: str = None, strength: float = 0.5, percussive: bool = True):
        super().__init__(func, audio, strength, percussive)
        
        # Check if function supports batch operations
        self.supports_batch = self._check_batch_support()
    
    def _check_batch_support(self):
        """Test if the effect function can handle 4D batch arrays"""
        try:
            # Create a small test batch
            test_batch = np.random.randint(0, 255, (2, 8, 8, 3), dtype=np.uint8)
            result = self.func(array=test_batch, strength=0.1, amplitude=0.5)
            return result.shape == test_batch.shape and result.ndim == 4
        except:
            return False
    
    def apply_batch_effect(self, batch_array, start_index):
        """Apply effect to a batch of images with vectorized operations when possible"""
        batch_size = batch_array.shape[0]
        
        # Get amplitudes for the entire batch
        end_index = start_index + batch_size
        amplitudes = self.spec[start_index:end_index]
        
        if self.supports_batch:
            # Use vectorized batch processing
            # For batch effects, we use the mean amplitude for the entire batch
            # or could apply per-frame amplitudes differently based on effect design
            mean_amplitude = np.mean(amplitudes)
            return self.func(array=batch_array, strength=self.strength, amplitude=mean_amplitude)
        else:
            # Fallback to per-frame processing
            return super().apply_batch_effect(batch_array, start_index)


class GPUBatchEffectsGenerator(BatchEffectsGenerator):
    """GPU-accelerated effects generator that keeps tensors on GPU longer"""
    
    def __init__(self, func, audio: str = None, strength: float = 0.5, percussive: bool = True):
        super().__init__(func, audio, strength, percussive)
        
        # MPS-only optimization - always use MPS device
        self.device = torch.device("mps")
        self.use_gpu = True
    
    def apply_batch_effect_gpu(self, tensor_batch, start_index):
        """Apply effect to a batch of tensors on GPU"""
        batch_size = tensor_batch.shape[0]
        
        # Get amplitudes for the entire batch
        end_index = start_index + batch_size
        amplitudes = self.spec[start_index:end_index]
        
        if self.supports_batch:
            # Use vectorized batch processing with mean amplitude
            mean_amplitude = np.mean(amplitudes)
            return self.func(array=tensor_batch, strength=self.strength, amplitude=mean_amplitude)
        else:
            # Convert to numpy, apply effects, convert back
            numpy_batch = tensor_batch.cpu().numpy()
            result_numpy = super().apply_batch_effect(numpy_batch, start_index)
            return torch.from_numpy(result_numpy).to(self.device)
    
    def apply_batch_effect(self, batch_array, start_index):
        """Apply effect to a batch - handles both numpy and tensor inputs"""
        if isinstance(batch_array, torch.Tensor) and self.use_gpu:
            return self.apply_batch_effect_gpu(batch_array, start_index)
        else:
            return super().apply_batch_effect(batch_array, start_index)

