"""
Vision Encoder for the Vision-to-Vision VisionTrader.

Uses a pretrained ViT to encode masked chart images into spatial
feature embeddings that the CNN decoder uses to generate the complete chart.
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import logging

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """
    Pretrained ViT encoder that processes 512×512 chart images.
    
    Outputs spatial patch embeddings (batch, num_patches, embed_dim)
    for use by the CNN decoder. Position embeddings are interpolated
    from the pretrained 384×384 resolution to 512×512.
    
    Args:
        model_name: HuggingFace ViT model name
        freeze_layers: Number of transformer layers to freeze (from the start)
        image_size: Input image resolution (H, W)
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch32-384",
        freeze_layers: int = 10,
        image_size: tuple = (512, 512),
    ):
        super().__init__()
        
        self.image_size = image_size
        self.freeze_layers = freeze_layers
        
        # Load pretrained ViT with custom image size
        # HuggingFace ViT automatically interpolates position embeddings
        config = ViTConfig.from_pretrained(model_name)
        config.image_size = image_size[0]  # Override to 512
        
        self.vit = ViTModel.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,  # Handles position embedding resize
        )
        
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_patches_per_side = image_size[0] // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Apply freeze strategy
        self._freeze_layers(freeze_layers)
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"VisionEncoder initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Image size: {image_size[0]}x{image_size[1]}")
        logger.info(f"  Patch size: {self.patch_size}")
        logger.info(f"  Patches: {self.num_patches_per_side}x{self.num_patches_per_side} = {self.num_patches}")
        logger.info(f"  Embed dim: {self.embed_dim}")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Trainable params: {trainable_params:,}")
        logger.info(f"  Frozen params: {frozen_params:,}")
    
    def _freeze_layers(self, num_layers: int):
        """Freeze embedding layer and first N transformer layers."""
        # Freeze patch embeddings
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of encoder layers
        for i, layer in enumerate(self.vit.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode chart image into spatial patch embeddings.
        
        Args:
            pixel_values: (batch, 3, H, W) normalized image tensor
            
        Returns:
            patch_embeddings: (batch, num_patches, embed_dim)
                Spatial features excluding CLS token.
        """
        outputs = self.vit(pixel_values=pixel_values)
        
        # outputs.last_hidden_state shape: (batch, num_patches + 1, embed_dim)
        # First token is CLS — drop it, keep only spatial patch embeddings
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        
        return patch_embeddings
    
    def get_spatial_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode and reshape to spatial grid for CNN decoder.
        
        Args:
            pixel_values: (batch, 3, H, W) normalized image tensor
            
        Returns:
            spatial_features: (batch, embed_dim, grid_h, grid_w)
        """
        patch_embeddings = self.forward(pixel_values)
        
        batch_size = patch_embeddings.shape[0]
        spatial = patch_embeddings.transpose(1, 2).reshape(
            batch_size,
            self.embed_dim,
            self.num_patches_per_side,
            self.num_patches_per_side,
        )
        
        return spatial
