"""
This module integrates speaker-specific x-vectors with log-Mel spectrograms. 

Key Components:
- `XVectorFusion`: Projects and fuses x-vectors with log-Mel spectrograms.
- `PersonalizedWhisper`: Integrates x-vector fusion into Whisper's encoder input.

Expected Output:
The fused feature representation retains the original mel-spectrogram dimensions, 
facilitating seamless integration into Whisper's processing pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class XVectorFusion(nn.Module):
    def __init__(self, xvec_dim, projection_dim, mel_dim):
        """
        xvec_dim: Dimension of the original x-vector (512)
        projection_dim: Dimension to project the x-vector to (64)
        mel_dim: Dimension of the log-Mel features (80)
        """
        super(XVectorFusion, self).__init__()
        # Project the x-vector from xvec_dim to projection_dim
        self.projection = nn.Linear(xvec_dim, projection_dim)
        self.activation = nn.ReLU()
        # Fusion layer: maps the concatenated features back to mel_dim
        self.fusion_linear = nn.Linear(mel_dim + projection_dim, mel_dim)

    
    def forward(self, mel_features, xvector):
        """
        mel_features: Tensor of shape [batch_size, T, mel_dim], the log-Mel features
        xvector: Tensor of shape [batch_size, xvec_dim], the pre-extracted x-vector
        """
        # Apply a linear projection followed by a ReLU to the x-vector
        projected_xvec = self.activation(self.projection(xvector))  # [batch_size, projection_dim]
        
        # Expand the projected x-vector along the time dimension
        batch_size, T, _ = mel_features.size()
        xvec_expanded = projected_xvec.unsqueeze(1).expand(batch_size, T, -1)  # [batch_size, T, projection_dim]
        
        # Concatenate the log-Mel features and the expanded x-vector along the feature dimension
        fused_features = torch.cat([mel_features, xvec_expanded], dim=-1)  # [batch_size, T, mel_dim + projection_dim]
        
        # Map the concatenated features back to the original mel_dim
        output = self.fusion_linear(fused_features)  # [batch_size, T, mel_dim]
        return output

class PersonalizedWhisper(nn.Module):
    def __init__(self, base_model, xvec_dim, projection_dim, mel_dim):
        """
        base_model: The original Whisper model (instance of WhisperForConditionalGeneration)
        xvec_dim: Original x-vector dimension (512)
        projection_dim: Projected dimension for x-vector (64)
        mel_dim: Dimension of the log-Mel features (e.g., 80)
        """
        super(PersonalizedWhisper, self).__init__()
        self.base_model = base_model
        # Fusion module to combine log-Mel features with x-vector information
        self.fusion_module = XVectorFusion(xvec_dim, projection_dim, mel_dim)
        self._keys_to_ignore_on_save = []
    
    def generate(self, *args, **kwargs):
        if "xvector" in kwargs:
            kwargs.pop("xvector")
        return self.base_model.generate(*args, **kwargs)
    
    def forward(self, input_features=None, xvector=None, **kwargs):
        """
        input_features: Tensor of shape [batch_size, T, mel_dim], the original log-Mel features
        ivector: Tensor of shape [batch_size, xvec_dim], the pre-extracted x-vector
        kwargs: Additional arguments to be passed to the base model (e.g., attention_mask, labels, etc.)
        """
        if xvector is None and "xvector" in kwargs:
            xvector = kwargs.pop("xvector")

        # If forgot both positional and kwarg 'xvector', provide fallback or error
        if xvector is None:
            raise ValueError("xvector is missing! Ensure data_collator or input dict includes 'xvector'.")

        # If 'input_features' was also not provided, either fallback or error
        if input_features is None and "input_features" in kwargs:
            input_features = kwargs.pop("input_features")

        if input_features is None:
            raise ValueError("input_features is missing! Ensure data_collator includes 'input_features'.")
        

        # Fuse x-vector + mel_features
        fused_features = self.fusion_module(input_features, xvector)

        # Now pass to base Whisper model
        return self.base_model(input_features=fused_features, **kwargs)
        
    @property
    def generation_config(self):
        """Expose the generation_config from the base model."""
        return self.base_model.generation_config

    @property
    def config(self):
        """Expose the config from the base model."""
        return self.base_model.config

    @property
    def tokenizer(self):
        """Expose the tokenizer from the base model if needed."""
        return self.base_model.tokenizer
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient_checkpointing_enable to the base model."""
        return self.base_model.gradient_checkpointing_enable(**kwargs)

batch_size = 4
T = 200
mel_dim = 80
xvec_dim = 512
projection_dim = 64

# Simulate log-Mel features and x-vectors from a data loader
mel_features = torch.randn(batch_size, T, mel_dim)
xvector = torch.randn(batch_size, xvec_dim)

fusion_module = XVectorFusion(xvec_dim, projection_dim, mel_dim)
fused_output = fusion_module(mel_features, xvector)
print("Fused output shape:", fused_output.shape)  # Expected output: [4, 200, 80]
