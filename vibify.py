#!/usr/bin/env python3
import sys
import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, VideoClip


def upscale_frame(frame, factor=2, method="lanczos"):
    """
    Upscale the input frame by the given factor with improved quality.

    Args:
        frame: Input frame to upscale
        factor: Upscaling factor (default: 2)
        method: Interpolation method - 'lanczos', 'cubic', or 'linear' (default: 'lanczos')
    """
    h, w = frame.shape[:2]

    # Select interpolation method
    if method == "lanczos":
        interpolation = cv2.INTER_LANCZOS4
    elif method == "cubic":
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR

    new_frame = cv2.resize(frame, (w * factor, h * factor), interpolation=interpolation)
    return new_frame


def apply_crt_warp(frame, strength=0.0005):
    """
    Apply a barrel distortion to mimic a curved CRT screen.
    """
    h, w = frame.shape[:2]
    # Generate normalized coordinate grids.
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    # Compute the radial distance from the center.
    r = np.sqrt(xv**2 + yv**2)
    # Apply a barrel distortion factor.
    factor = 1 + strength * (r**2)
    x_new = xv * factor
    y_new = yv * factor
    # Map the new coordinates back to the image coordinate space.
    map_x = ((x_new + 1) * (w - 1) / 2).astype(np.float32)
    map_y = ((y_new + 1) * (h - 1) / 2).astype(np.float32)
    # Remap the frame using the new coordinate mappings.
    warped = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def apply_color_grade(frame):
    """
    Adjust the colors to boost neon-like tones.
    Expects and returns float32 frame in [0, 1] range.
    """
    # Apply separate scaling factors to each color channel.
    r = np.clip(frame[:, :, 0] * 1.2, 0, 1)
    g = np.clip(frame[:, :, 1] * 1.0, 0, 1)
    b = np.clip(frame[:, :, 2] * 1.3, 0, 1)
    # Reassemble the channels into a single frame.
    new_frame = np.stack([r, g, b], axis=2)
    # Apply a gamma correction for a smoother tonal transition.
    gamma = 1.1
    new_frame = np.power(new_frame, gamma)
    return new_frame


def apply_chromatic_aberration(frame, shift=2):
    """
    Slightly shift the red and blue channels to create a chromatic aberration effect.
    Expects and returns float32 frame in [0, 1] range.
    """
    h, w = frame.shape[:2]

    # Copy original channels
    red = frame[:, :, 0].copy()
    blue = frame[:, :, 2].copy()
    green = frame[:, :, 1]

    # Shift red channel to the right, fill edges with edge pixels
    if shift > 0:
        red[:, shift:] = frame[:, :-shift, 0]
        red[:, :shift] = frame[:, 0:1, 0]  # Fill left edge

    # Shift blue channel to the left, fill edges with edge pixels
    if shift > 0:
        blue[:, :-shift] = frame[:, shift:, 2]
        blue[:, -shift:] = frame[:, -1:, 2]  # Fill right edge

    # Combine the channels back into one frame.
    aberrated = np.stack((red, green, blue), axis=2)
    return aberrated


def apply_scanlines(frame, intensity=0.1):
    """
    Overlay horizontal scanlines by darkening alternating rows.
    Simple alternating pattern for clear visibility.
    Expects and returns float32 frame in [0, 1] range.
    """
    h, w = frame.shape[:2]

    # Create a simple alternating scanline mask
    mask = np.ones((h, 1), dtype=np.float32)
    # Darken every other row (simple alternating pattern)
    mask[1::2] = 1.0 - intensity

    # Repeat the mask across the width of the frame.
    mask = np.repeat(mask, w, axis=1)
    # Stack the mask to match the 3 color channels.
    mask = np.stack([mask] * 3, axis=2)
    # Apply the mask to the frame.
    scanlined = frame * mask
    return scanlined


def add_noise(frame, noise_level=0.03):
    """
    Add film grain-style noise to emulate analog signal imperfections.
    Expects and returns float32 frame in [0, 1] range.
    """
    h, w = frame.shape[:2]

    # Generate film grain with proper frequency characteristics
    # Create multiple noise layers with different frequencies
    base_noise = np.random.randn(h, w) * noise_level
    fine_noise = np.random.randn(h, w) * (noise_level * 0.5)

    # Combine noise layers
    combined_noise = base_noise + fine_noise

    # Apply noise with luminance-dependent intensity
    # (film grain is more visible in darker areas)
    luminance = np.mean(frame, axis=2, keepdims=True)
    noise_intensity = 1.0 - (luminance * 0.3)  # Less noise in bright areas

    # Stack noise across color channels with slight variations
    noise_r = combined_noise * noise_intensity[:, :, 0]
    noise_g = combined_noise * noise_intensity[:, :, 0] * 0.9
    noise_b = combined_noise * noise_intensity[:, :, 0] * 1.1

    noise_3d = np.stack([noise_r, noise_g, noise_b], axis=2)

    # Add noise to the frame.
    noisy = frame + noise_3d
    # Clip the values to ensure they remain valid pixel intensities.
    noisy = np.clip(noisy, 0, 1)
    return noisy


def process_frame(frame):
    """
    Upscale the frame and chain together all processing steps.
    Optimized order to minimize artifacts and maintain quality.
    """
    # Convert to float32 [0, 1] range for better precision
    frame = frame.astype(np.float32) / 255.0

    # Apply color grading early to preserve color relationships
    frame = apply_color_grade(frame)

    # Upscale the enhanced frame to higher resolution
    frame = upscale_frame((frame * 255).astype(np.uint8), factor=2)
    frame = frame.astype(np.float32) / 255.0

    # Apply scanlines last for straight lines (simulates viewing screen)
    frame = apply_scanlines(frame, intensity=0.1)

    # Apply chromatic aberration (happens in electron gun)
    frame = apply_chromatic_aberration(frame, shift=4)

    # Apply spatial distortions (curved glass effect) - VERY strong
    frame = apply_crt_warp((frame * 255).astype(np.uint8), strength=0.01)
    frame = frame.astype(np.float32) / 255.0

    # Add noise last to simulate final analog imperfections
    frame = add_noise(frame, noise_level=0.03)

    # Convert back to uint8 for output
    return np.clip(frame * 255, 0, 255).astype(np.uint8)


def linear_interpolate_clip(clip):
    """
    Create a new clip by linearly interpolating between consecutive frames,
    effectively doubling the frame rate.
    """

    def make_frame(t):
        F = clip.fps
        frame_duration = 1.0 / F
        # Determine the surrounding frame times
        t1 = (int(t * F)) * frame_duration
        t2 = t1 + frame_duration
        # If t2 exceeds the clip's duration, just return the last frame.
        if t2 > clip.duration:
            return clip.get_frame(t1)
        # Compute the interpolation weight.
        weight = (t - t1) / frame_duration
        frame1 = clip.get_frame(t1).astype(np.float32)
        frame2 = clip.get_frame(t2).astype(np.float32)
        frame_interp = (1 - weight) * frame1 + weight * frame2
        return np.clip(frame_interp, 0, 255).astype(np.uint8)

    new_clip = VideoClip(make_frame, duration=clip.duration)
    new_clip.fps = clip.fps * 2  # Double the frame rate
    return new_clip


def main(input_file):
    # Load the video clip.
    clip = VideoFileClip(input_file)

    # Apply linear frame interpolation to double the frame rate.
    interpolated_clip = linear_interpolate_clip(clip)

    # Process each frame with the defined pipeline.
    processed_clip = interpolated_clip.fl_image(process_frame)

    # Reapply the original audio BEFORE writing the video
    processed_clip = processed_clip.set_audio(clip.audio)

    # Construct the output file name from original basename
    original_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{original_basename}_vibed.mp4"

    # Write the processed video to a new file WITH audio.
    processed_clip.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        audio=True,
        fps=processed_clip.fps,  # maintain correct fps
        ffmpeg_params=["-crf", "18", "-preset", "slow"],  # High quality encoding
    )

    # Return the processed clip with audio (optional)
    return processed_clip


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vibify.py <input_video_file>")
        sys.exit(1)
    main(sys.argv[1])
