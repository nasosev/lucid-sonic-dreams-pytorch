import time
import os
from lucidsonicdreams import LucidSonicDream
from lucidsonicdreams.main import EffectsGenerator

# Prevents "RuntimeError: MPS backend out of memory" errors on MacOS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

filename = "models/lhq-256-stylegan3-t-25Mimg.pkl"


if __name__ == "__main__":
    start_time = time.time()  # Start tracking time

    L = LucidSonicDream(song="song.mp3", style=filename)
    L.hallucinate(
        file_name="song.mp4",
        resolution=256,
        fps=24,
        contrast_strength=1,
        flash_strength=1,
    )

    end_time = time.time()  # End tracking time
    total_time = end_time - start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds")
