#!/usr/bin/env python3

"""
Simple, clean layer extraction for psychedelic visuals
Mathematically elegant approach using PCA for dimensionality reduction
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

# Prevent MPS memory issues
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from lucidsonicdreams import LucidSonicDream

def reduce_channels_pca(tensor, n_components=3):
    """
    Clean PCA-based dimensionality reduction
    Maps high-dimensional channels to 3D RGB space
    """
    if tensor.ndim != 4:
        return torch.zeros(1, 3, 256, 256, device=tensor.device)
    
    batch, channels, height, width = tensor.shape
    
    # Reshape for PCA: [pixels, channels]
    reshaped = tensor[0].permute(1, 2, 0).reshape(-1, channels).cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped)
    
    # Reshape back: [height, width, n_components]
    reduced = reduced.reshape(height, width, n_components)
    
    # Convert to tensor and add batch dimension: [1, n_components, height, width]
    rgb_tensor = torch.from_numpy(reduced).permute(2, 0, 1).unsqueeze(0).float().to(tensor.device)
    
    # Normalize each channel to [0, 1]
    for i in range(n_components):
        channel = rgb_tensor[:, i:i+1, :, :]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        rgb_tensor[:, i:i+1, :, :] = channel
    
    # Scale to StyleGAN range [-1, 1]
    rgb_tensor = rgb_tensor * 2.0 - 1.0
    
    # Resize to 256x256 if needed
    if rgb_tensor.shape[-1] != 256:
        rgb_tensor = F.interpolate(rgb_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    
    return rgb_tensor

# Store original functions
original_hallucinate = LucidSonicDream.hallucinate
original_generate_frames = LucidSonicDream.generate_frames

def simple_patched_hallucinate(self, capture_layer=None, **kwargs):
    """Simple layer capture with PCA reduction"""
    self._capture_layer = capture_layer
    if capture_layer:
        print(f"🎨 Capturing layer: {capture_layer}")
        print("   Using PCA for clean dimensionality reduction")
    return original_hallucinate(self, **kwargs)

def simple_patched_generate_frames(self):
    """Generate frames with simple layer extraction"""
    
    if hasattr(self, '_capture_layer') and self._capture_layer and hasattr(self, 'Gs'):
        print(f"🎬 Generating layer visualization: {self._capture_layer}")
        
        original_forward = self.Gs.synthesis.forward
        
        def replacement_forward(*args, **kwargs):
            captured_output = None
            
            def hook_fn(module, input, output):
                nonlocal captured_output
                captured_output = output.clone() if isinstance(output, torch.Tensor) else output
            
            # Find target layer
            target_module = None
            for name, module in self.Gs.synthesis.named_modules():
                if name == self._capture_layer:
                    target_module = module
                    break
            
            if target_module is None:
                print(f"❌ Layer {self._capture_layer} not found!")
                return original_forward(*args, **kwargs)
            
            handle = target_module.register_forward_hook(hook_fn)
            
            try:
                # Run original forward pass
                _ = original_forward(*args, **kwargs)
                
                if captured_output is not None:
                    # Simple PCA reduction
                    processed = reduce_channels_pca(captured_output)
                    return processed
                else:
                    print(f"❌ Failed to capture {self._capture_layer}")
                    return original_forward(*args, **kwargs)
                    
            finally:
                handle.remove()
        
        # Replace synthesis forward
        self.Gs.synthesis.forward = replacement_forward
        
        try:
            return original_generate_frames(self)
        finally:
            self.Gs.synthesis.forward = original_forward
    else:
        return original_generate_frames(self)

# Apply patches
LucidSonicDream.hallucinate = simple_patched_hallucinate
LucidSonicDream.generate_frames = simple_patched_generate_frames

print("✅ Simple layer extraction patch applied!")
print()
print("🎨 Clean layer visualization using PCA dimensionality reduction")
print("   - Captures raw layer outputs")
print("   - Maps high-dimensional channels to RGB using PCA")
print("   - Mathematically elegant and simple")
print()
print("Usage: python test.py sample.mp3 L13_256_128")