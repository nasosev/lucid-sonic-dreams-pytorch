#!/usr/bin/env python3
import sys
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, VideoClip


def upscale_frame(frame, factor=2):
    """
    Upscale the input frame by the given factor.
    """
    h, w = frame.shape[:2]
    # Resize the frame using linear interpolation to achieve a smooth upscale.
    new_frame = cv2.resize(
        frame, (w * factor, h * factor), interpolation=cv2.INTER_LINEAR
    )
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
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def apply_color_grade(frame):
    """
    Adjust the colors to boost neon-like tones.
    """
    # Normalize the frame values to [0, 1] for float precision.
    frame_float = frame.astype(np.float32) / 255.0
    # Apply separate scaling factors to each color channel.
    r = np.clip(frame_float[:, :, 0] * 1.2, 0, 1)
    g = np.clip(frame_float[:, :, 1] * 1.0, 0, 1)
    b = np.clip(frame_float[:, :, 2] * 1.3, 0, 1)
    # Reassemble the channels into a single frame.
    new_frame = np.stack([r, g, b], axis=2)
    # Apply a gamma correction for a smoother tonal transition.
    gamma = 1.1
    new_frame = np.power(new_frame, gamma)
    # Convert back to 8-bit color depth.
    new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
    return new_frame


def apply_chromatic_aberration(frame, shift=2):
    """
    Slightly shift the red and blue channels to create a chromatic aberration effect.
    """
    # Shift the red channel to the right.
    red = np.roll(frame[:, :, 0], shift, axis=1)
    # Shift the blue channel to the left.
    blue = np.roll(frame[:, :, 2], -shift, axis=1)
    # The green channel remains unchanged.
    green = frame[:, :, 1]
    # Combine the channels back into one frame.
    aberrated = np.stack((red, green, blue), axis=2)
    return aberrated


def apply_scanlines(frame, intensity=0.1):
    """
    Overlay horizontal scanlines by darkening alternating rows.
    This version uses vectorized slicing for efficiency.
    """
    h, w = frame.shape[:2]
    # Create a mask that darkens every other row.
    mask = np.ones((h, 1), dtype=np.float32)
    mask[1::2] = 1.0 - intensity  # Darken every other row
    # Repeat the mask across the width of the frame.
    mask = np.repeat(mask, w, axis=1)
    # Stack the mask to match the 3 color channels.
    mask = np.stack([mask] * 3, axis=2)
    # Apply the mask to the frame.
    scanlined = (frame.astype(np.float32) * mask).astype(np.uint8)
    return scanlined


def add_noise(frame, noise_level=0.03):
    """
    Add subtle random noise to emulate analog signal imperfections.
    """
    # Generate Gaussian noise based on the specified noise level.
    noise = np.random.randn(*frame.shape) * (255 * noise_level)
    # Add noise to the frame.
    noisy = frame.astype(np.float32) + noise
    # Clip the values to ensure they remain valid pixel intensities.
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def process_frame(frame):
    """
    Upscale the frame and chain together all processing steps.
    """
    # Upscale the frame to a higher resolution
    frame = upscale_frame(frame, factor=2)
    # Apply CRT warp to mimic a curved CRT display.
    frame = apply_crt_warp(frame, strength=0.0005)
    # Adjust the colors for a neon-like appearance.
    frame = apply_color_grade(frame)
    # Apply chromatic aberration to shift color channels.
    frame = apply_chromatic_aberration(frame, shift=2)
    # Overlay horizontal scanlines for an analog effect.
    frame = apply_scanlines(frame, intensity=0.1)
    # Add subtle noise to simulate analog imperfections.
    frame = add_noise(frame, noise_level=0.03)
    return frame


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

    # Construct the output file name.
    output_file = input_file.rsplit(".", 1)[0] + "_vibed.mp4"

    # Write the processed video to a new file WITH audio.
    processed_clip.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        audio=True,
        fps=processed_clip.fps,  # maintain correct fps
    )

    # Return the processed clip with audio (optional)
    return processed_clip


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vibify.py <input_video_file>")
        sys.exit(1)
    main(sys.argv[1])
