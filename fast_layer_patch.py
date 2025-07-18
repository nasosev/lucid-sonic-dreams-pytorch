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

def apply_pca_reordering(tensor, pca_order="rbg"):
    """Apply PCA component reordering based on specified order"""
    if pca_order == "rbg":
        return tensor[:, [0, 2, 1], :, :]  # Default order: [0,2,1] -> [R,B,G]
    if pca_order == "rgb":
        return tensor  # Natural order: [0,1,2] -> [R,G,B]
    
    # Create mapping from letter to component index
    order_map = {'r': 0, 'g': 1, 'b': 2}
    
    # Validate order string
    if len(pca_order) != 3 or set(pca_order) != {'r', 'g', 'b'}:
        print(f"Warning: Invalid PCA order '{pca_order}', using default 'rgb'")
        return tensor
    
    # Create reordering indices
    indices = [order_map[c] for c in pca_order]
    
    # Reorder channels: tensor is [batch, n_components, height, width]
    return tensor[:, indices, :, :]

# Global variable to store fixed PCA components
_fixed_pca_model = None

def reduce_channels_pca_fast(tensor, n_components=3, use_fixed_pca=False):
    """Fast PCA reduction - optimized for performance"""
    if tensor.ndim != 4:
        return None
    
    batch, channels, height, width = tensor.shape
    
    # Collect all batch data for per-batch PCA fitting (preserve spatial structure)
    all_batch_data = []
    for b in range(batch):
        reshaped = tensor[b].permute(1, 2, 0).reshape(-1, channels).cpu().numpy()
        all_batch_data.append(reshaped)
    
    # Use fixed PCA model if available and requested
    global _fixed_pca_model
    if use_fixed_pca and _fixed_pca_model is not None:
        pca = _fixed_pca_model
    else:
        # Concatenate all batch data and fit single PCA model for consistency
        combined_data = np.concatenate(all_batch_data, axis=0)  # [total_pixels, channels]
        pca = PCA(n_components=n_components)
        pca.fit(combined_data)
        
        # Store the PCA model if this is the first time and fixed PCA is requested
        if use_fixed_pca and _fixed_pca_model is None:
            _fixed_pca_model = pca
    
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
    
    # Apply PCA reordering if specified
    if 'PCA_ORDER' in globals():
        rgb_tensor = apply_pca_reordering(rgb_tensor, globals()['PCA_ORDER'])
    
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
        use_fixed_pca = getattr(synthesis, '_use_fixed_pca', False)
        processed = reduce_channels_pca_fast(x, use_fixed_pca=use_fixed_pca)
        if processed is None:
            # ws is already unbound, need to stack it back
            ws_stacked = torch.stack(ws, dim=1)
            return original_forward(ws_stacked, **layer_kwargs)
        return processed
    
    return early_stopping_forward

# Store original functions
original_hallucinate = LucidSonicDream.hallucinate
original_generate_frames = LucidSonicDream.generate_frames

def fast_patched_hallucinate(self, capture_layer=None, use_fixed_pca=False, **kwargs):
    """Fast layer capture"""
    self._capture_layer = capture_layer
    self._use_fixed_pca = use_fixed_pca
    if capture_layer:
        print(f"🚀 Fast layer capture: {capture_layer}")
        if use_fixed_pca:
            print(f"🎨 Using fixed PCA components from first frame")
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
        
        # Set the fixed PCA flag on the synthesis object
        self.Gs.synthesis._use_fixed_pca = getattr(self, '_use_fixed_pca', False)
        
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