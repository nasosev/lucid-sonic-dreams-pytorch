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

def reduce_channels_pca_fast(tensor, n_components=3, use_fixed_pca=True):
    """PCA reduction optimized for single frame processing"""
    from sklearn.decomposition import PCA
    
    if tensor.ndim != 4:
        return None
    
    # Single frame processing (batch_size=1)
    channels, height, width = tensor.shape[1:]
    device = tensor.device
    
    global _fixed_pca_model
    
    # Use cached components (all frames after first)
    if _fixed_pca_model is not None:
        components = _fixed_pca_model['components'].to(device)
        mean = _fixed_pca_model['mean'].to(device)
        
        data = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, channels).float()
        reduced_data = (data - mean) @ components.T
    else:
        # First time: high-quality CPU PCA
        data = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, channels).cpu().numpy()
        pca = PCA(n_components=n_components)
        pca.fit(data)
        reduced_data = pca.transform(data)
        
        # Cache components for all subsequent frames
        _fixed_pca_model = {
            'components': torch.from_numpy(pca.components_).float(),
            'mean': torch.from_numpy(pca.mean_).float()
        }
        
        reduced_data = torch.from_numpy(reduced_data).to(device)
    
    # Reshape to single frame
    rgb_tensor = reduced_data.reshape(height, width, n_components).permute(2, 0, 1).unsqueeze(0)
    
    if 'PCA_ORDER' in globals():
        rgb_tensor = apply_pca_reordering(rgb_tensor, globals()['PCA_ORDER'])
    
    # Simple normalization for single frame
    rgb_flat = rgb_tensor.view(1, n_components, -1)
    mins = rgb_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
    maxs = rgb_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
    range_vals = torch.where(maxs - mins < 1e-8, torch.ones_like(maxs), maxs - mins)
    
    return ((rgb_tensor - mins) / range_vals) * 2.0 - 1.0


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
        """Modified forward pass that stops at target layer - WITH LAYER PROFILING"""
        import time
        
        # Convert to the expected format
        ws = ws.to(torch.float32).unbind(dim=1)
        
        # Get the synthesis object from the method
        synthesis = original_forward.__self__
        
        # LAYER PROFILING: Input layer timing
        input_start = time.time()
        x = synthesis.input(ws[0])
        input_time = time.time() - input_start
        
        # Find target layer index
        target_index = get_layer_index(synthesis, target_layer)
        if target_index is None:
            # ws is already unbound, need to stack it back
            ws_stacked = torch.stack(ws, dim=1)
            return original_forward(ws_stacked, **layer_kwargs)
        
        # LAYER PROFILING: Track each layer's timing
        layer_times = []
        
        # Execute layers up to and including target
        for idx, (name, w) in enumerate(zip(synthesis.layer_names, ws[1:])):
            layer_start = time.time()
            x = getattr(synthesis, name)(x, w, **layer_kwargs)
            layer_time = time.time() - layer_start
            layer_times.append((name, layer_time))
            
            # Stop immediately after target layer
            if idx == target_index:
                break
        
        # Apply output scaling if needed
        if hasattr(synthesis, 'output_scale') and synthesis.output_scale != 1:
            x = x * synthesis.output_scale
        
        # LAYER PROFILING: Print detailed breakdown every 10 frames to reduce spam
        # Check if verbose mode is enabled (passed through from main instance)
        verbose_enabled = getattr(synthesis, '_verbose_enabled', False)
        
        if verbose_enabled:
            if not hasattr(synthesis, '_profile_counter'):
                synthesis._profile_counter = 0
            
            if synthesis._profile_counter % 10 == 0:
                total_layer_time = sum(t for _, t in layer_times)
                print(f"\n🔍 LAYER-BY-LAYER SYNTHESIS TIMING (Frame {synthesis._profile_counter}, Early Stopping at {target_layer}):")
                print(f"  Input layer: {input_time:.3f}s")
                for name, layer_time in layer_times:
                    pct = (layer_time / total_layer_time) * 100 if total_layer_time > 0 else 0
                    print(f"  {name}: {layer_time:.3f}s ({pct:.1f}%)")
                print(f"  Total synthesis: {input_time + total_layer_time:.3f}s")
                print()
                
            synthesis._profile_counter += 1
        
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
        print(f"🚀 Fast layer capture: {capture_layer} (fixed PCA)")
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
        
        # Set verbose mode on the synthesis object
        self.Gs.synthesis._verbose_enabled = getattr(self, 'verbose', False)
        
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