#!/usr/bin/env python3
import sys
import numpy as np
import cv2
from moviepy.editor import VideoFileClip


def apply_crt_warp(frame, strength=0.0005):
    """
    Apply a barrel distortion to mimic a curved CRT screen.
    """
    h, w = frame.shape[:2]
    # Create a normalized coordinate grid
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv * xv + yv * yv)
    # Compute new coordinates with a slight barrel effect
    factor = 1 + strength * (r**2)
    x_new = xv * factor
    y_new = yv * factor
    # Map normalized coordinates back to pixel positions
    map_x = ((x_new + 1) * (w - 1) / 2).astype(np.float32)
    map_y = ((y_new + 1) * (h - 1) / 2).astype(np.float32)
    warped = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def apply_color_grade(frame):
    """
    Adjust the colors to boost neon-like tones.
    """
    # Convert to float and scale to [0,1]
    frame_float = frame.astype(np.float32) / 255.0
    # Boost the red and blue channels for a pinkish/cyan look
    r = np.clip(frame_float[:, :, 0] * 1.2, 0, 1)
    g = np.clip(frame_float[:, :, 1] * 1.0, 0, 1)
    b = np.clip(frame_float[:, :, 2] * 1.3, 0, 1)
    new_frame = np.stack([r, g, b], axis=2)
    # Apply a slight gamma correction to further adjust contrast
    gamma = 1.1
    new_frame = np.power(new_frame, gamma)
    new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
    return new_frame


def apply_chromatic_aberration(frame, shift=2):
    """
    Slightly shift the red and blue channels to create a chromatic aberration effect.
    """
    red = np.roll(frame[:, :, 0], shift, axis=1)
    blue = np.roll(frame[:, :, 2], -shift, axis=1)
    # Green channel remains unshifted
    green = frame[:, :, 1]
    aberrated = np.stack((red, green, blue), axis=2)
    return aberrated


def apply_scanlines(frame, intensity=0.1):
    """
    Overlay horizontal scanlines by darkening alternating rows.
    """
    h, w = frame.shape[:2]
    # Create a mask that dims every other row
    mask = np.ones((h, 1), dtype=np.float32)
    for i in range(h):
        if i % 2 != 0:
            mask[i, 0] = 1.0 - intensity
    mask = np.repeat(mask, w, axis=1)
    mask = np.stack([mask] * 3, axis=2)
    scanlined = frame.astype(np.float32) * mask
    scanlined = np.clip(scanlined, 0, 255).astype(np.uint8)
    return scanlined


def add_noise(frame, noise_level=0.03):
    """
    Add subtle random noise to emulate analog signal imperfections.
    """
    noise = np.random.randn(*frame.shape) * (255 * noise_level)
    noisy = frame.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def process_frame(frame):
    """
    Chain together all processing steps on a single frame.
    """
    # Note: MoviePy uses RGB order.
    frame = apply_crt_warp(frame, strength=0.0005)
    frame = apply_color_grade(frame)
    frame = apply_chromatic_aberration(frame, shift=2)
    frame = apply_scanlines(frame, intensity=0.1)
    frame = add_noise(frame, noise_level=0.03)
    return frame


def main(input_file):
    # Load the video clip
    clip = VideoFileClip(input_file)
    # Process each frame
    processed_clip = clip.fl_image(process_frame)
    # Define output filename
    output_file = input_file.rsplit(".", 1)[0] + "_vibed.mp4"
    # Write the processed video to file using H.264 encoding
    processed_clip.write_videofile(output_file, codec="libx264")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vibify.py <input_video_file>")
        sys.exit(1)
    main(sys.argv[1])
