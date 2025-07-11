import time
import os
import sys
import numpy as np
from scipy.stats import truncnorm
import pickle

from lucidsonicdreams import LucidSonicDream
from lucidsonicdreams.main import EffectsGenerator

# Prevent "RuntimeError: MPS backend out of memory" errors on MacOS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Define latent space dimension (typically 512 for StyleGAN models)
latent_dim = 512


def generate_random_latent_center(seed: int = None):
    """
    Generate a random latent center using a truncated normal distribution.
    If a seed is provided, the random generator is seeded for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
    return truncnorm.rvs(-2, 2, size=(latent_dim,)).astype(np.float32)


def save_latent_vector(latent_vector, filename):
    """
    Save the latent vector to a file so it can be reloaded later.
    """
    with open(filename, "wb") as f:
        pickle.dump(latent_vector, f)


if __name__ == "__main__":
    # Get input filename from command line argument or use default
    if len(sys.argv) > 1:
        input_audio = sys.argv[1]
        print(f"Using input audio file: {input_audio}")
    else:
        input_audio = "sample.mp3"
        print(f"No input file specified, using default: {input_audio}")
    
    # Ask user if they want to use latent vector constraints
    use_latent = input("Use latent vector constraints? (y/n): ").strip().lower() == "y"

    latent_center = None
    latent_radius = None
    seed_value = None

    if use_latent:
        # Prompt for a seed value to generate a reproducible latent vector.
        seed_input = input(
            "Enter a seed for generating the latent vector (or press Enter for random): "
        ).strip()
        seed_value = int(seed_input) if seed_input else None

        # Generate and save the latent center
        latent_center = generate_random_latent_center(seed=seed_value)
        latent_radius = 10  # Adjust this value as needed
        latent_filename = (
            f"latent_vector_seed_{seed_value}.pkl"
            if seed_value is not None
            else "latent_vector_random.pkl"
        )
        save_latent_vector(latent_center, latent_filename)
        print(f"Latent vector saved to {latent_filename}")
    else:
        # Ask for seed even when not using latent constraints for reproducibility
        seed_input = input(
            "Enter a seed for reproducible generation (or press Enter for random): "
        ).strip()
        seed_value = int(seed_input) if seed_input else None
        if seed_value is not None:
            print(f"Using seed: {seed_value}")

    # Specify the style weights file (ensure this path is correct)
    filename = "models/lhq-256-stylegan3-t-25Mimg.pkl"

    start_time = time.time()  # Start tracking time

    # Instantiate LucidSonicDream with optional latent center, radius, and seed
    L = LucidSonicDream(
        song=input_audio,
        style=filename,
        latent_center=latent_center,
        latent_radius=latent_radius,
        seed=seed_value,
    )
    L.hallucinate(
        file_name="song.mp4",
        resolution=256,
        fps=24,
        speed_fpm=3,  # Even slower scene changes (default is 12)
        pulse_react=0.3,  # Gentler pulse reactions (default is 0.5)
        motion_react=0.3,  # Gentler motion reactions (default is 0.5)
        class_smooth_seconds=3,  # Smoother class transitions (default is 1)
        motion_randomness=0.3,  # Less random motion (default is 0.5)
        class_pitch_react=0.3,  # Gentler class reactions (default is 0.5)
        contrast_strength=0.5,
        flash_strength=0.5,
        save_frames=True,
    )

    end_time = time.time()  # End tracking time
    total_time = end_time - start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds")
