"""
CNN Decoder for the Vision-to-Vision VisionTrader.
Reconstructs the full chart image from spatial feature embeddings.
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ChartDecoder(nn.Module):
    """
    CNN-based decoder that upsamples spatial features to a full image.
    
    Architecture:
    - Input: (B, 768, 16, 16) from VisionEncoder
    - Projection: 768 -> Initial Channels
    - Upsampling Blocks: 5x (Upsample -> Conv -> BN -> ReLU)
    - Output: (B, 3, 512, 512)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        channels: list = [512, 256, 128, 64, 32],
        output_channels: int = 3,
    ):
        super().__init__()
        
        self.channels = channels
        
        # 1. Projection from Encoder Embed Dim to first decoder channel
        self.projection = nn.Sequential(
            nn.Conv2d(embed_dim, channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # 2. Upsampling Blocks
        # We need 5 blocks to go from 16x16 to 512x512 (16 * 2^5 = 512)
        layers = []
        
        # Block 1: 16 -> 32
        layers.append(self._make_block(channels[0], channels[1]))
        
        # Block 2: 32 -> 64
        layers.append(self._make_block(channels[1], channels[2]))
        
        # Block 3: 64 -> 128
        layers.append(self._make_block(channels[2], channels[3]))
        
        # Block 4: 128 -> 256
        layers.append(self._make_block(channels[3], channels[4]))
        
        # Block 5: 256 -> 512
        # Re-use last channel dim if list is exhausted, or map to a smaller dim
        # Here we map 32 -> 32 to keep it lightweight before final projection
        layers.append(self._make_block(channels[4], channels[4]))
        
        self.upsampling = nn.Sequential(*layers)
        
        # 3. Final Prediction
        self.final_layer = nn.Sequential(
            nn.Conv2d(channels[4], output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output is [0, 1] for image
        )
        
        self._log_architecture()

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a single upsampling block.
        Using Nearest Neighbor + Conv instead of ConvTranspose to avoid checkerboard artifacts.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _log_architecture(self):
        params = sum(p.numel() for p in self.parameters())
        logger.info(f"ChartDecoder initialized:")
        logger.info(f"  Input dim: 768")
        logger.info(f"  Channel sequence: {self.channels} -> {self.channels[-1]}")
        logger.info(f"  Total parameters: {params:,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 768, 16, 16) spatial features
        Returns:
            x: (B, 3, 512, 512) reconstructed image
        """
        x = self.projection(x)
        x = self.upsampling(x)
        x = self.final_layer(x)
        return x
