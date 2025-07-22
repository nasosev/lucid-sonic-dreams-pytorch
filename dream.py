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
    # Show examples if no arguments provided
    if len(sys.argv) == 1:
        print("🎨 Lucid Sonic Dreams - Psychedelic Layer Visualization")
        print()
        print("Usage: python dream.py [audio_file] [options]")
        print()
        print("Examples:")
        print("  python dream.py                                    # Use defaults")
        print("  python dream.py my_song.mp3                        # Use custom audio")
        print(
            "  python dream.py --layer L12_276_128                # Psychedelic layer"
        )
        print(
            "  python dream.py my_song.mp3 --layer L8_276_645 --seed 42  # Full control"
        )
        print()
        print("Defaults: audio=sample.wav, model=lhq-256, layer=final, seed=random")
        print()
        print("Add --help for full options")
        print()

    # Parse all arguments properly with argparse
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate psychedelic videos from audio using StyleGAN layer extraction"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="sample.wav",
        help="Input audio file (default: sample.wav)",
    )
    parser.add_argument(
        "--layer", help="StyleGAN layer to extract (default: final layer)"
    )
    parser.add_argument(
        "--model",
        default="lhq-256-stylegan3-t-25Mimg.pkl",
        help="StyleGAN model file (default: lhq-256-stylegan3-t-25Mimg.pkl)",
    )
    parser.add_argument(
        "--latent", action="store_true", help="Use latent vector constraints"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible generation (default: random)",
    )
    parser.add_argument(
        "--pca",
        default="rbg",
        help="PCA component order for RGB channels (default: rbg). Examples: rgb, grb, gbr, brg, bgr",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed timing and profiling logs",
    )

    args = parser.parse_args()

    # Get audio file
    input_audio = args.audio_file
    print(f"Using input audio file: {input_audio}")

    # Get layer capture argument (optional)
    capture_layer = args.layer
    pca_order = args.pca.lower()
    if capture_layer:
        print(f"Using layer capture: {capture_layer}")
        print(f"Using PCA component order: {pca_order}")
        # Set PCA order globally for the patch
        globals()["PCA_ORDER"] = pca_order
        # Add the fast layer extraction patch (optimized for performance)
        exec(open("fast_layer_patch.py").read())

    # Get model file (with default)
    filename = f"models/{args.model}"
    if args.model == "lhq-256-stylegan3-t-25Mimg.pkl":
        print(f"Using default model: {args.model}")
    else:
        print(f"Using model: {args.model}")

    # Check latent vector flag
    use_latent = args.latent
    if use_latent:
        print("🎯 Using latent vector constraints for exploration")

    latent_center = None
    latent_radius = None
    seed_value = args.seed

    if use_latent:
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

    if seed_value is not None:
        print(f"Using seed: {seed_value}")

    # Model file was set above from command line arguments

    start_time = time.time()  # Start tracking time

    # Instantiate LucidSonicDream with optional latent center, radius, and seed
    L = LucidSonicDream(
        song=input_audio,
        style=filename,
        latent_center=latent_center,
        latent_radius=latent_radius,
        seed=seed_value,
        verbose=args.verbose,
    )

    # Generate output filename from input audio file
    audio_basename = os.path.splitext(os.path.basename(input_audio))[0]
    output_filename = f"{audio_basename}.mp4"

    # Build hallucinate parameters
    hallucinate_params = {
        "file_name": output_filename,
        "fps": 24,
        "speed_fpm": 6,  # Even slower scene changes (default is 12)
        "pulse_react": 0.6,  # Stronger pulse reactions (default is 0.5)
        "motion_react": 0.1,  # Gentler motion reactions (default is 0.5)
        "contrast_strength": 0.2,
        "flash_strength": 0.2,
        "save_frames": True,
    }

    # Add layer capture if specified
    if capture_layer:
        hallucinate_params["capture_layer"] = capture_layer
        print(f"🎨 Generating psychedelic video with layer: {capture_layer}")

    L.hallucinate(**hallucinate_params)

    end_time = time.time()  # End tracking time
    total_time = end_time - start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds")
