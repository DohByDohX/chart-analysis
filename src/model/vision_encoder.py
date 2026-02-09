import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionEncoder(nn.Module):
    """
    Vision Transformer Encoder for 512x512 chart images.
    Uses a pre-trained ViT model with interpolated position embeddings.
    """
    
    def __init__(self, model_name: str = "google/vit-base-patch32-384"):
        """
        Initialize the Vision Encoder.
        
        Args:
            model_name: HuggingFace model hub name
        """
        super().__init__()
        
        logger.info(f"Loading pre-trained ViT model: {model_name}")
        
        # Load pre-trained model
        # We use interpolate_pos_encoding=True in the forward pass usually, 
        # but modern transformers handles it via configuration or automatic detection
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Freezing option (can be exposed later if needed)
        # for param in self.vit.parameters():
        #     param.requires_grad = False
            
        self.embed_dim = self.vit.config.hidden_size
        logger.info(f"Vision Encoder initialized. Embed Dim: {self.embed_dim}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            Sequence of embeddings (B, SeqLen, EmbedDim)
        """
        # HuggingFace ViT expects pixel_values argument
        # interpolate_pos_encoding=True tells it to resize embeddings for 512x512 input
        outputs = self.vit(pixel_values=images, interpolate_pos_encoding=True)
        
        # last_hidden_state shape: (Batch, SeqLen, HiddenDim)
        return outputs.last_hidden_state
