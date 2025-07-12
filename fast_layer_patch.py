#!/usr/bin/env python3

"""
Fast layer extraction - optimized for performance
Minimal logging, optimized PCA, reduced GPU-CPU transfers
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

# Prevent MPS memory issues
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from lucidsonicdreams import LucidSonicDream

def reduce_channels_pca_fast(tensor, n_components=3):
    """Fast PCA reduction - optimized for performance"""
    if tensor.ndim != 4:
        return None
    
    batch, channels, height, width = tensor.shape
    
    # Special handling for 3-channel layers (already RGB-like)
    if channels == 3:
        return tensor.clone()
    
    # Collect all batch data for per-batch PCA fitting (preserve spatial structure)
    all_batch_data = []
    for b in range(batch):
        reshaped = tensor[b].permute(1, 2, 0).reshape(-1, channels).cpu().numpy()
        all_batch_data.append(reshaped)
    
    # Concatenate all batch data and fit single PCA model for consistency
    combined_data = np.concatenate(all_batch_data, axis=0)  # [total_pixels, channels]
    pca = PCA(n_components=n_components)
    pca.fit(combined_data)
    
    # Apply the same PCA transform to each batch item
    batch_results = []
    for b in range(batch):
        # Transform using the shared PCA model
        reduced = pca.transform(all_batch_data[b])
        reduced = reduced.reshape(height, width, n_components)
        
        # Convert to tensor: [n_components, height, width]
        item_tensor = torch.from_numpy(reduced).permute(2, 0, 1).float().to(tensor.device)
        batch_results.append(item_tensor)
    
    # Stack batch results: [batch, n_components, height, width]
    rgb_tensor = torch.stack(batch_results, dim=0)
    
    # Fast vectorized normalization (all channels at once)
    rgb_flat = rgb_tensor.view(batch, n_components, -1)  # [batch, channels, pixels]
    mins = rgb_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)  # [batch, channels, 1, 1]
    maxs = rgb_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)  # [batch, channels, 1, 1]
    
    rgb_tensor = (rgb_tensor - mins) / (maxs - mins + 1e-8)
    rgb_tensor = rgb_tensor * 2.0 - 1.0
    
    return rgb_tensor

def get_layer_index(synthesis, target_layer):
    """Get the index of target layer in layer_names list"""
    if not hasattr(synthesis, 'layer_names'):
        return None
    
    try:
        return synthesis.layer_names.index(target_layer)
    except ValueError:
        return None

def create_early_stopping_forward(original_forward, target_layer):
    """Create early stopping version of synthesis forward pass"""
    
    def early_stopping_forward(ws, **layer_kwargs):
        """Modified forward pass that stops at target layer"""
        
        # Convert to the expected format
        ws = ws.to(torch.float32).unbind(dim=1)
        
        # Get the synthesis object from the method
        synthesis = original_forward.__self__
        
        # Start with input layer
        x = synthesis.input(ws[0])
        
        # Find target layer index
        target_index = get_layer_index(synthesis, target_layer)
        if target_index is None:
            # ws is already unbound, need to stack it back
            ws_stacked = torch.stack(ws, dim=1)
            return original_forward(ws_stacked, **layer_kwargs)
        
        # Execute layers up to and including target (minimal logging)
        for idx, (name, w) in enumerate(zip(synthesis.layer_names, ws[1:])):
            x = getattr(synthesis, name)(x, w, **layer_kwargs)
            
            # Stop immediately after target layer
            if idx == target_index:
                break
        
        # Apply output scaling if needed
        if hasattr(synthesis, 'output_scale') and synthesis.output_scale != 1:
            x = x * synthesis.output_scale
        
        # Process the captured output (minimal logging)
        processed = reduce_channels_pca_fast(x)
        if processed is None:
            # ws is already unbound, need to stack it back
            ws_stacked = torch.stack(ws, dim=1)
            return original_forward(ws_stacked, **layer_kwargs)
        return processed
    
    return early_stopping_forward

# Store original functions
original_hallucinate = LucidSonicDream.hallucinate
original_generate_frames = LucidSonicDream.generate_frames

def fast_patched_hallucinate(self, capture_layer=None, **kwargs):
    """Fast layer capture"""
    self._capture_layer = capture_layer
    if capture_layer:
        print(f"🚀 Fast layer capture: {capture_layer}")
    return original_hallucinate(self, **kwargs)

def fast_patched_generate_frames(self):
    """Fast frame generation"""
    
    if hasattr(self, '_capture_layer') and self._capture_layer and hasattr(self, 'Gs'):
        target_layer = self._capture_layer
        
        # Store original synthesis forward
        original_synthesis_forward = self.Gs.synthesis.forward
        
        # Create fast early stopping version
        early_stopping_forward = create_early_stopping_forward(
            original_synthesis_forward, 
            target_layer
        )
        
        # Replace synthesis forward method
        self.Gs.synthesis.forward = early_stopping_forward
        
        try:
            return original_generate_frames(self)
        finally:
            # Always restore original forward method
            self.Gs.synthesis.forward = original_synthesis_forward
    else:
        return original_generate_frames(self)

# Apply patches
LucidSonicDream.hallucinate = fast_patched_hallucinate
LucidSonicDream.generate_frames = fast_patched_generate_frames

print("🚀 Fast layer extraction patch applied!")
print("   - Minimal logging for maximum speed")
print("   - Optimized PCA computation")  
print("   - Reduced GPU-CPU transfers")
print("   - Vectorized normalization")