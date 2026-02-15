"""
VisionTrader: End-to-End Vision-to-Vision Model.
Combines VisionEncoder (ViT) and ChartDecoder (CNN).
"""
import torch
import torch.nn as nn
import logging
from .encoder import VisionEncoder
from .decoder import ChartDecoder

logger = logging.getLogger(__name__)


class VisionTrader(nn.Module):
    """
    Vision-to-Vision model for chart prediction.
    
    Flow:
    1. Input: Masked chart image (B, 3, 512, 512)
    2. Encoder: ViT extracts spatial features (B, 768, 16, 16)
    3. Decoder: CNN reconstructs full image (B, 3, 512, 512)
    """
    
    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch32-384",
        freeze_encoder_layers: int = 10,
        encoder_embed_dim: int = 768,
        decoder_channels: list = [512, 256, 128, 64, 32],
        image_size: tuple = (512, 512),
    ):
        super().__init__()
        
        self.image_size = image_size
        
        # 1. Vision Encoder
        self.encoder = VisionEncoder(
            model_name=vit_model_name,
            freeze_layers=freeze_encoder_layers,
            image_size=image_size
        )
        
        # 2. Chart Decoder
        self.decoder = ChartDecoder(
            embed_dim=encoder_embed_dim,
            channels=decoder_channels,
            output_channels=3  # RGB
        )
        
        self._log_model_info()
        
    def _log_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"VisionTrader initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 512, 512) normalized input images
        Returns:
            reconstructed: (B, 3, 512, 512) predicted images
        """
        # 1. Encode to spatial features
        # Shape: (B, 768, 16, 16)
        features = self.encoder.get_spatial_features(x)
        
        # 2. Decode to full image
        # Shape: (B, 3, 512, 512)
        reconstructed = self.decoder(features)
        
        return reconstructed
