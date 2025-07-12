#!/usr/bin/env python3

"""
Safe efficient layer extraction - no recursion issues
Simple hook-based approach that captures early and exits
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

# Prevent MPS memory issues
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from lucidsonicdreams import LucidSonicDream

def reduce_channels_pca(tensor, target_size=256, n_components=3):
    """Clean PCA reduction with upscaling"""
    if tensor.ndim != 4:
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            # Affine layer visualization
            params = tensor[0].cpu().numpy()
            pattern = np.zeros((target_size, target_size, 3))
            
            x = np.linspace(0, 1, target_size)
            y = np.linspace(0, 1, target_size)
            xx, yy = np.meshgrid(x, y)
            
            n_params = min(len(params), 12)
            for i in range(min(3, n_params)):
                if i < n_params:
                    if i == 0:
                        pattern[:, :, 0] = np.sin(xx * params[i] * 10) * np.cos(yy * params[i] * 10)
                    elif i == 1:
                        pattern[:, :, 1] = np.sin(xx * params[i] * 8 + params[i]) * np.cos(yy * params[i] * 12)
                    elif i == 2:
                        pattern[:, :, 2] = np.sin(xx * params[i] * 6) + np.cos(yy * params[i] * 8)
            
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
            rgb_tensor = torch.from_numpy(pattern).permute(2, 0, 1).unsqueeze(0).float().to(tensor.device)
            return rgb_tensor * 2.0 - 1.0
        
        return torch.zeros(1, 3, target_size, target_size, device=tensor.device)
    
    batch, channels, height, width = tensor.shape
    print(f"🎨 Processing {channels} channels at {height}x{width}")
    
    # PCA reduction
    reshaped = tensor[0].permute(1, 2, 0).reshape(-1, channels).cpu().numpy()
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped)
    reduced = reduced.reshape(height, width, n_components)
    
    rgb_tensor = torch.from_numpy(reduced).permute(2, 0, 1).unsqueeze(0).float().to(tensor.device)
    
    # Normalize
    for i in range(n_components):
        channel = rgb_tensor[:, i:i+1, :, :]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        rgb_tensor[:, i:i+1, :, :] = channel
    
    rgb_tensor = rgb_tensor * 2.0 - 1.0
    
    # Upscale if needed
    if rgb_tensor.shape[-1] != target_size:
        rgb_tensor = F.interpolate(rgb_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
        print(f"   Upscaled to {target_size}x{target_size}")
    
    return rgb_tensor

# Global storage for captured output
_captured_output = None
_target_layer = None

def capture_hook(module, input, output):
    """Simple hook to capture layer output"""
    global _captured_output
    _captured_output = output.clone() if isinstance(output, torch.Tensor) else output

# Store original functions
original_hallucinate = LucidSonicDream.hallucinate
original_generate_frames = LucidSonicDream.generate_frames

def safe_patched_hallucinate(self, capture_layer=None, **kwargs):
    """Safe layer capture"""
    global _target_layer
    _target_layer = capture_layer
    if capture_layer:
        print(f"🎨 Safe layer capture: {capture_layer}")
    return original_hallucinate(self, **kwargs)

def safe_patched_generate_frames(self):
    """Safe frame generation with layer capture"""
    global _captured_output, _target_layer
    
    if _target_layer and hasattr(self, 'Gs'):
        print(f"🎬 Generating with layer capture: {_target_layer}")
        
        # Find target module
        target_module = None
        for name, module in self.Gs.synthesis.named_modules():
            if name == _target_layer:
                target_module = module
                break
        
        if target_module is None:
            print(f"❌ Layer {_target_layer} not found!")
            return original_generate_frames(self)
        
        # Hook the target layer
        handle = target_module.register_forward_hook(capture_hook)
        
        # Store original synthesis forward
        original_synthesis_forward = self.Gs.synthesis.forward
        
        def efficient_synthesis_forward(ws, **kwargs):
            """Modified synthesis that uses captured layer output"""
            global _captured_output
            _captured_output = None
            
            # Run original synthesis (this will trigger our hook)
            original_result = original_synthesis_forward(ws, **kwargs)
            
            # If we captured something, use it instead
            if _captured_output is not None:
                print(f"✅ Captured layer output: {_captured_output.shape}")
                processed = reduce_channels_pca(_captured_output, target_size=kwargs.get('img_resolution', 256))
                return processed
            else:
                return original_result
        
        # Replace synthesis forward
        self.Gs.synthesis.forward = efficient_synthesis_forward
        
        try:
            result = original_generate_frames(self)
            return result
        finally:
            # Restore everything
            handle.remove()
            self.Gs.synthesis.forward = original_synthesis_forward
    else:
        return original_generate_frames(self)

# Apply patches
LucidSonicDream.hallucinate = safe_patched_hallucinate
LucidSonicDream.generate_frames = safe_patched_generate_frames

print("✅ Safe efficient layer patch applied!")
print()
print("🎨 Simple hook-based layer capture")
print("   - No recursion issues")
print("   - Clean PCA reduction")
print("   - Safe module handling")
print()
print("Usage: python test.py sample.mp3 L12_276_128")