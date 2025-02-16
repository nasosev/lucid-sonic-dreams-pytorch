# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
import os
import re
from typing import List, Optional, Tuple, Union
import sys

import click
import numpy as np
import PIL.Image
import torch
from lucidsonicdreams import LucidSonicDream


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

models = [
    "Abstract art",
    "Anime portraits",
    "CIFAR 10",
    "CIFAR 100",
    "Doors",
    "Maps",
    "Visionary Art",
    "WikiArt",
    "beetles",
    "cakes",
    "car (config-e)",
    "car (config-f)",
    "cat",
    "church",
    "faces (FFHQ config-e 256x256)",
    "faces (FFHQ config-e)",
    "faces (FFHQ config-f 512x512)",
    "faces (FFHQ config-f)",
    "faces (FFHQ slim 256x256)",
    "figure drawings",
    "flowers",
    "fursona",
    "grumpy cat",
    "horse",
    "microscope images",
    "modern art",
    "my little pony",
    "obama",
    "painting faces",
    "panda",
    "textures",
    "trypophobia",
    "ukiyoe faces",
    "wildlife",
]
output_path = '/tmp'


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        model_type: str = Input(
            description="Which checkpoint to use?",
            default="Visionary Art",
            choices=models,
        ),
        audio_file: Path = Input(
            description="Path to the uploaded audio file (.mp3, .wav are supported)"
        ),
        fps: int = Input(
            description="Frames per second of generated video", default=20
        ),
        pulse_react: int = Input(
            description="The 'strength' of the pulse. It is recommended to keep this between 0 and 100.",
            default=60,
        ),
        pulse_react_to: str = Input(
            description="Whether the pulse should react to percussive or harmonic elements",
            choices=["percussive", "harmonic"],
            default="percussive",
        ),
        motion_react: int = Input(
            description="The 'strength' of the motion. It is recommended to keep this between 0 and 100.",
            default=60,
        ),
        motion_react_to: str = Input(
            description="Whether the motion should react to percussive or harmonic elements",
            choices=["percussive", "harmonic"],
            default="harmonic",
        ),
        motion_randomness: int = Input(
            description="Degree of randomness of motion. Higher values will typically prevent the video from cycling through the same visuals repeatedly. Must range from 0 to 100.",
            default=50,
        ),
        truncation: int = Input(
            description='Controls the variety of visuals generated. Lower values lead to lower variety. Note: A very low value will usually lead to "jittery" visuals. Must range from 0 to 100.',
            default=50,
        ),
        batch_size: int = Input(
            description="The number of images to generate at once. Higher values will typically lead to faster generation, but may cause memory issues. Must range from 1 to 100.",
            default=10,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        pulse_percussive = pulse_react_to == "percussive"
        pulse_harmonic = pulse_react_to == "harmonic"

        motion_percussive = motion_react_to == "percussive"
        motion_harmonic =  motion_react_to == "harmonic"

        L = LucidSonicDream(song = audio_file,
                            style = f"/src/models/{model_type}.pkl")

        L.hallucinate(file_name = f"{output_path}/lucid-sonic-dream.mp4",
              fps = fps,
              motion_percussive = motion_percussive,
              motion_harmonic = motion_harmonic,
              pulse_percussive = pulse_percussive,
              pulse_harmonic = pulse_harmonic,
              pulse_react = pulse_react / 100,
              motion_react = motion_react / 100,
              motion_randomness = motion_randomness / 100,
              truncation = truncation / 100,
              start = 0,
              batch_size=batch_size,)
        return Path(f"{output_path}/lucid-sonic-dream.mp4")

# if __name__ == "__main__":
#     model = Predictor()
#     response = model.predict(audio_file="https://download.samplelib.com/mp3/sample-6s.mp3")
