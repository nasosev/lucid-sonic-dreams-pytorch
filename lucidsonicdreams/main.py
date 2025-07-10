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
import skimage.exposure
import librosa
import soundfile
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip
import pygit2
from importlib import import_module

from .helper_functions import *
from .sample_effects import *

import cv2

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


class MultiTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __getitem__(self, i):
        return [t[i] for t in self.tensor_list]

    def __len__(self):
        return len(self.tensor_list[0])


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
            device = torch.device("mps" if torch.mps.is_available() else "cpu")
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
        """Load normalized spectrograms and chromagram"""

        start = self.start
        duration = self.duration
        fps = self.fps
        input_shape = self.input_shape
        pulse_percussive = self.pulse_percussive
        pulse_harmonic = self.pulse_harmonic
        motion_percussive = self.motion_percussive
        motion_harmonic = self.motion_harmonic

        # Load audio signal data
        wav, sr = librosa.load(self.song, offset=start, duration=duration)
        wav_motion = wav_pulse = wav_class = wav
        sr_motion = sr_pulse = sr_class = sr

        # If pulse_percussive != pulse_harmonic
        # or motion_percussive != motion_harmonic,
        # decompose harmonic and percussive signals and assign accordingly
        aud_unassigned = (not self.pulse_audio) or (not self.motion_audio)
        pulse_bools_equal = pulse_percussive == pulse_harmonic
        motion_bools_equal = motion_percussive == motion_harmonic

        if aud_unassigned and not all([pulse_bools_equal, motion_bools_equal]):
            wav_harm, wav_perc = librosa.effects.hpss(wav)
            wav_list = [wav, wav_harm, wav_perc]

            pulse_bools = [pulse_bools_equal, pulse_harmonic, pulse_percussive]
            wav_pulse = wav_list[pulse_bools.index(max(pulse_bools))]

            motion_bools = [motion_bools_equal, motion_harmonic, motion_percussive]
            wav_motion = wav_list[motion_bools.index(max(motion_bools))]

        # Load audio signal data for Pulse, Motion, and Class if provided
        if self.pulse_audio:
            wav_pulse, sr_pulse = librosa.load(
                self.pulse_audio, offset=start, duration=duration
            )
        if self.motion_audio:
            wav_motion, sr_motion = librosa.load(
                self.motion_audio, offset=start, duration=duration
            )
        if self.class_audio:
            wav_class, sr_class = librosa.load(
                self.class_audio, offset=start, duration=duration
            )

        # Calculate frame duration (i.e. samples per frame)
        frame_duration = int(sr / fps - (sr / fps % 64))

        # Generate normalized spectrograms for Pulse, Motion and Class
        self.spec_norm_pulse = get_spec_norm(
            wav_pulse, sr_pulse, input_shape, frame_duration
        )
        self.spec_norm_motion = get_spec_norm(
            wav_motion, sr_motion, input_shape, frame_duration
        )
        self.spec_norm_class = get_spec_norm(
            wav_class, sr_class, input_shape, frame_duration
        )

        # Generate chromagram from Class audio
        chrom_class = librosa.feature.chroma_cqt(
            y=wav_class, sr=sr, hop_length=frame_duration
        )
        # Sort pitches based on "dominance"
        chrom_class_norm = chrom_class / chrom_class.sum(axis=0, keepdims=1)
        chrom_class_sum = np.sum(chrom_class_norm, axis=1)
        pitches_sorted = np.argsort(chrom_class_sum)[::-1]

        # Assign attributes to be used for vector generation
        self.wav, self.sr, self.frame_duration = wav, sr, frame_duration
        self.chrom_class, self.pitches_sorted = chrom_class, pitches_sorted

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
        if len(class_vecs) == 0:

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
        if type(class_shuffle_seconds) == int:
            if frame != 0 and frame % round(class_shuffle_seconds * fps) == 0:
                return True
            else:
                return False

        # If class_shuffle_seconds is a list, return True if current timestamp
        # (in seconds) is in list
        if type(class_shuffle_seconds) == list:
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

        # Initialize lists for incremental updates and class vectors
        pulse_noise = []
        motion_noise = []
        self.class_vecs = []

        # Base vectors for Pulse and Motion updates
        pulse_base = np.array([self.pulse_react] * self.input_shape)
        motion_base = np.array([motion_react] * self.input_shape)

        # Initialize update directions and randomness factors
        self.motion_signs = np.array(
            [random.choice([1, -1]) for _ in range(self.input_shape)]
        )
        rand_factors = np.where(
            np.random.rand(self.input_shape) < 0.5, 1, 1 - self.motion_randomness
        )
        cumulative_motion = np.zeros(self.input_shape, dtype=np.float32)

        for i in range(num_frames):
            # Reinitialize randomness factors every 4 seconds
            if i % round(fps * 4) == 0:
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

            pulse_noise.append(pulse_noise_add)
            motion_noise.append(motion_noise_add)

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
            self.class_vecs.append(class_vec_add)

        # Smooth class vectors by averaging over frames and interpolating
        if class_smooth_frames > 1:
            class_frames_interp = [
                np.mean(self.class_vecs[i : i + class_smooth_frames], axis=0)
                for i in range(0, len(self.class_vecs), class_smooth_frames)
            ]
            self.class_vecs = full_frame_interpolation(
                class_frames_interp, class_smooth_frames, len(self.class_vecs)
            )

        # Convert lists to numpy arrays for downstream processing
        self.noise = np.array(self.noise)
        self.class_vecs = np.array(self.class_vecs)

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

            contrast = EffectsGenerator(
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

            flash = EffectsGenerator(
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
            image_batch = queue.get()
            if image_batch is None:
                queue.task_done()
                break
            if image_batch.ndim == 3:
                array = image_batch
                image_index = i * self.batch_size
                for effect in self.custom_effects:
                    array = effect.apply_effect(array=array, index=image_index)

                #   # Save. Include leading zeros in file name to keep alphabetical order
                max_frame_index = num_frame_batches * self.batch_size + self.batch_size
                file_name = str(image_index).zfill(len(str(max_frame_index)))
                self.file_names.append(file_name)
                self.final_images.append(array)
            else:
                for j, array in enumerate(image_batch):
                    image_index = (i * self.batch_size) + j

                    #   # Apply efects
                    for effect in self.custom_effects:
                        array = effect.apply_effect(array=array, index=image_index)

                    #   # Save. Include leading zeros in file name to keep alphabetical order
                    max_frame_index = (
                        num_frame_batches * self.batch_size + self.batch_size
                    )
                    file_name = str(image_index).zfill(len(str(max_frame_index)))
                    self.file_names.append(file_name)
                    self.final_images.append(array)
            i += 1
            queue.task_done()

        # def generate_frames(self):
        """Generate GAN output for each frame of video"""

        file_name = self.file_name
        resolution = self.resolution
        batch_size = self.batch_size
        if self.use_tf:
            Gs_syn_kwargs = {
                "output_transform": {
                    "func": self.convert_images_to_uint8,
                    "nchw_to_nhwc": True,
                },
                "randomize_noise": False,
                "minibatch_size": batch_size,
            }
        else:
            Gs_syn_kwargs = {"noise_mode": "const"}  # random, const, None

        # Set-up temporary frame directory
        self.frames_dir = file_name.split(".mp4")[0] + "_frames"
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)

        # create dataloader
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

        print("Using device:", device)
        ds = MultiTensorDataset(
            [torch.from_numpy(self.noise), torch.from_numpy(self.class_vecs)]
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4
        )

        # executor = concurrent.futures.ProcessPoolExecutor()

        self.final_images = []
        self.file_names = []
        # Generate frames
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, (noise_batch, class_batch) in enumerate(
                tqdm(dl, position=0, desc="Generating frames")
            ):
                # If style is a custom function, pass batches to the function
                if callable(self.style):
                    image_batch = self.style(
                        noise_batch=noise_batch, class_batch=class_batch
                    )
                # Otherwise, generate frames with StyleGAN
                else:
                    if self.use_tf:
                        noise_batch = noise_batch.numpy()
                        class_batch = class_batch.numpy()
                        w_batch = self.Gs.components.mapping.run(
                            noise_batch, np.tile(class_batch, (batch_size, 1))
                        )
                        image_batch = self.Gs.components.synthesis.run(
                            w_batch, **Gs_syn_kwargs
                        )
                        image_batch = np.array(image_batch)
                    else:
                        noise_batch = noise_batch.to(device=device, dtype=torch.float32)
                        # Convert once using torch.tensor, ensuring the target device is specified
                        noise_tensor = torch.tensor(
                            noise_batch, device=device, dtype=torch.float32
                        )
                        class_tensor = torch.tensor(
                            class_batch, device=device, dtype=torch.float32
                        )
                        with torch.no_grad():
                            w_batch = self.Gs.mapping(
                                noise_tensor,
                                class_tensor,
                                truncation_psi=self.truncation_psi,
                            )
                            image_batch = self.Gs.synthesis(w_batch, **Gs_syn_kwargs)
                        image_ready = (
                            (image_batch.permute(0, 2, 3, 1) * 127.5 + 128)
                            .clamp(0, 255)
                            .to(torch.uint8)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )
                        if batch_size == 1:
                            image_ready = np.expand_dims(image_ready, axis=0)
                        future_to_url = {
                            executor.submit(
                                self.images_saver,
                                image_ready,
                                batch_size * i,
                                resolution,
                            )
                        }

    def generate_frames(self):
        """Generate GAN output for each frame of video."""
        file_name = self.file_name
        resolution = self.resolution
        batch_size = self.batch_size

        # Set synthesis kwargs based on framework choice
        if self.use_tf:
            Gs_syn_kwargs = {
                "output_transform": {
                    "func": self.convert_images_to_uint8,
                    "nchw_to_nhwc": True,
                },
                "randomize_noise": False,
                "minibatch_size": batch_size,
            }
        else:
            Gs_syn_kwargs = {"noise_mode": "const"}  # Options: random, const, None

        # Set up temporary frame directory
        self.frames_dir = file_name.split(".mp4")[0] + "_frames"
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)

        # Create dataloader from noise and class vectors
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        print("Using device:", device)
        ds = MultiTensorDataset(
            [torch.from_numpy(self.noise), torch.from_numpy(self.class_vecs)]
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4
        )

        self.final_images = []
        self.file_names = []

        # Use a ThreadPoolExecutor to handle image saving concurrently
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, (noise_batch, class_batch) in enumerate(
                tqdm(dl, position=0, desc="Generating frames")
            ):
                # Generate images via custom style function or StyleGAN
                if callable(self.style):
                    image_batch = self.style(
                        noise_batch=noise_batch, class_batch=class_batch
                    )
                else:
                    if self.use_tf:
                        # Convert tensor batches to numpy arrays for TF processing
                        noise_batch_np = noise_batch.numpy()
                        class_batch_np = class_batch.numpy()
                        # Tile class batch if necessary (ensuring shape compatibility)
                        tiled_class = np.tile(class_batch_np, (batch_size, 1))
                        w_batch = self.Gs.components.mapping.run(
                            noise_batch_np, tiled_class
                        )
                        image_batch = self.Gs.components.synthesis.run(
                            w_batch, **Gs_syn_kwargs
                        )
                        image_batch = np.array(image_batch)
                    else:
                        # For PyTorch: move to device and run synthesis with no_grad
                        # Convert once using torch.tensor, ensuring the target device is specified
                        noise_tensor = torch.tensor(
                            noise_batch, device=device, dtype=torch.float32
                        )
                        class_tensor = torch.tensor(
                            class_batch, device=device, dtype=torch.float32
                        )
                        with torch.no_grad():
                            w_batch = self.Gs.mapping(
                                noise_tensor,
                                class_tensor,
                                truncation_psi=self.truncation_psi,
                            )
                            image_batch = self.Gs.synthesis(w_batch, **Gs_syn_kwargs)
                        # Convert tensor output to uint8 numpy array
                        image_batch = (
                            (image_batch.permute(0, 2, 3, 1) * 127.5 + 128)
                            .clamp(0, 255)
                            .to(torch.uint8)
                            .cpu()
                            .numpy()
                        )

                # Ensure image_batch has shape (batch_size, height, width, channels)
                if batch_size == 1 and image_batch.ndim == 3:
                    image_batch = np.expand_dims(image_batch, axis=0)

                # Submit the image saving task to the executor
                future = executor.submit(
                    self.images_saver, image_batch, batch_size * i, resolution
                )
                futures.append(future)

            # Wait for all image-saving tasks to complete
            concurrent.futures.wait(futures)

    def images_saver(self, image, current_counter: int, resolution):
        file_names = []
        final_images = []
        for j, array in enumerate(image):
            file_names.append(str(current_counter + j).zfill(len(str(len(self.noise)))))
            for effect in self.custom_effects:
                array = effect.apply_effect(array=array, index=current_counter + j)
            final_images.append(array)
        for file_name, final_image in zip(
            file_names, final_images
        ):  # tqdm(zip(file_names, final_images), position=1, leave=False, desc="Storing frames", total=len(file_names)):
            with Image.fromarray(final_image, mode="RGB") as final_image_PIL:
                # If resolution is provided, resize
                if resolution:
                    final_image_PIL = final_image_PIL.resize((resolution, resolution))
                final_image_PIL.save(os.path.join(self.frames_dir, file_name + ".tiff"))

    def hallucinate(
        self,
        file_name: str,
        output_audio: str = None,
        fps: int = 43,
        resolution: int = None,
        start: float = 0,
        duration: float = None,
        save_frames: bool = False,
        batch_size: int = 1,
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
        self.batch_size = batch_size
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

        # Initialize style
        if not self.style_exists:
            print("Preparing style...")
            if not callable(self.style):
                self.stylegan_init()
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

        # Get normalized spectrogram
        self.spec = get_spec_norm(wav, sr, n_mels=n_mels, hop_length=hop_length)

    def apply_effect(self, array, index):
        """Apply effect to image (array)"""

        amplitude = self.spec[index]
        return self.func(array=array, strength=self.strength, amplitude=amplitude)
