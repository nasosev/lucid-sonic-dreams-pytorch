#!/usr/bin/env python3

"""
Script to explore available layers in the StyleGAN3 model
This will help identify the penultimate layer name
"""

import sys
sys.path.append('stylegan3')
import torch
import pickle

def explore_model_layers(model_file):
    # Load the model
    print(f"Loading StyleGAN3 model: {model_file}")
    with open(f'models/{model_file}', 'rb') as f:
        data = pickle.load(f)
        G = data['G_ema']
    
    print(f"Model loaded successfully!")
    print(f"Resolution: {G.img_resolution}")
    print(f"Number of layers: {G.synthesis.num_layers}")
    
    print("\nAvailable synthesis layers:")
    layer_names = []
    for name, module in G.synthesis.named_modules():
        if name and name.startswith('L') and '_' in name:
            layer_names.append(name)
            print(f"  {name}")
    
    if layer_names:
        print(f"\nFinal layer: {layer_names[-1]}")
        if len(layer_names) > 1:
            print(f"Penultimate layer: {layer_names[-2]}")
        else:
            print("Only one layer found")
    
    return layer_names

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        print("Usage: python explore_layers.py <model_file>")
        print("Example: python explore_layers.py stylegan3-r-afhqv2-512x512.pkl")
        sys.exit(1)
    
    try:
        layers = explore_model_layers(model_file)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the StyleGAN3 model downloaded and torch is available")