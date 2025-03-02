import os
import shutil
import time
import numpy as np
from scipy.stats import truncnorm
from multiprocessing import freeze_support

from lucidsonicdreams import LucidSonicDream


start = 999
end = 1000

# Define latent space dimension (typically 512 for StyleGAN models)
latent_dim = 512


def generate_random_latent_center(seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    return truncnorm.rvs(-2, 2, size=(latent_dim,)).astype(np.float32)


def main():
    # Folder to collect first frames from each seed
    consolidated_folder = "seed-images"
    if not os.path.exists(consolidated_folder):
        os.makedirs(consolidated_folder)

    # Folder for full outputs (each seed may create its own frames subfolder)
    output_dir = "seed_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_file = "song.mp3"

    # Process a range of seeds
    for seed in range(start, end):
        print(f"Processing seed: {seed}")
        start_time = time.time()

        latent_center = generate_random_latent_center(seed=seed)
        latent_radius = 0.5  # Adjust if needed

        video_file = os.path.join(output_dir, f"seed_{seed}.mp4")

        # Create the LucidSonicDream instance with the seed-generated latent center
        L = LucidSonicDream(
            song=audio_file,
            style="models/lhq-256-stylegan3-t-25Mimg.pkl",
            latent_center=latent_center,
            latent_radius=latent_radius,
        )

        # Call hallucinate to generate a very short output (1 frame)
        # fps=1 and duration=1 yields one frame; save_frames=True forces saving to a frames folder.
        L.hallucinate(
            file_name=video_file,
            fps=1,
            duration=1,
            save_frames=True,
            contrast_strength=1,
            flash_strength=1,
        )

        elapsed = time.time() - start_time
        print(
            f"Seed {seed} processed in {elapsed:.2f} seconds. Video saved to {video_file}"
        )

        # Locate the frames folder (e.g. "seed_0_frames") from the hallucinate process.
        if hasattr(L, "frames_dir") and os.path.exists(L.frames_dir):
            # List all files (we assume frames are saved as .tiff or .png)
            frame_files = [
                f for f in os.listdir(L.frames_dir) if f.endswith((".tiff", ".png"))
            ]
            if frame_files:
                # Sort the filenames and pick the first one (lowest sorted order)
                first_frame = sorted(frame_files)[0]
                src_path = os.path.join(L.frames_dir, first_frame)
                dest_path = os.path.join(
                    consolidated_folder,
                    f"seed_{seed}_first{os.path.splitext(first_frame)[1]}",
                )
                shutil.copy(src_path, dest_path)
                print(f"Copied first frame for seed {seed} to {dest_path}")
            else:
                print(
                    f"Warning: No frame files found in {L.frames_dir} for seed {seed}"
                )
        else:
            print(f"Warning: frames_dir not found for seed {seed}")


if __name__ == "__main__":
    freeze_support()  # Ensures proper bootstrapping on macOS
    main()
