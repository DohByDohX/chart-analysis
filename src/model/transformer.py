import torch
import torch.nn as nn
from typing import Optional
import logging

from .vision_encoder import VisionEncoder
from .token_decoder import TokenDecoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionTrader(nn.Module):
    """
    Complete Vision-to-Token Transformer model.
    Combines Vision Encoder (ViT) and Token Decoder for candlestick prediction.
    """
    
    def __init__(
        self,
        vit_model_name: str,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 16
    ):
        """
        Initialize VisionTrader.
        
        Args:
            vit_model_name: HuggingFace ViT model name
            vocab_size: Size of token vocabulary
            embed_dim: Embedding dimension (must match ViT output)
            num_heads: Number of attention heads in decoder
            num_layers: Number of decoder layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for generation
        """
        super().__init__()
        
        # Vision Encoder
        self.encoder = VisionEncoder(model_name=vit_model_name)
        
        # Token Decoder
        self.decoder = TokenDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        self.vocab_size = vocab_size
        logger.info(f"VisionTrader initialized. Vocab: {vocab_size}, Embed: {embed_dim}")
    
    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Training forward pass.
        
        Args:
            images: Batch of chart images (B, 3, H, W)
            tgt_tokens: Target token sequence (B, SeqLen)
            
        Returns:
            Logits (B, SeqLen, VocabSize)
        """
        # Encode images
        memory = self.encoder(images)  # (B, 257, 768)
        
        # Decode tokens
        logits = self.decoder(tgt_tokens, memory)  # (B, SeqLen, VocabSize)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        start_token: int,
        max_len: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressively generate token sequences.
        
        Args:
            images: Batch of chart images (B, 3, H, W)
            start_token: Starting token ID
            max_len: Maximum generation length
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, sample from top-k tokens only
            
        Returns:
            Generated tokens (B, max_len)
        """
        self.eval()
        
        batch_size = images.size(0)
        device = images.device
        
        # Encode images once
        memory = self.encoder(images)  # (B, 257, 768)
        
        # Initialize with start token
        generated = torch.full(
            (batch_size, 1),
            start_token,
            dtype=torch.long,
            device=device
        )
        
        # Autoregressive generation
        for _ in range(max_len - 1):
            # Get logits for current sequence
            logits = self.decoder(generated, memory)  # (B, SeqLen, VocabSize)
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature  # (B, VocabSize)
            
            # Optional top-k sampling
            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, indices, values)
            
            # Sample next token (greedy if temperature=1.0 and no top_k)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    @torch.no_grad()
    def generate_greedy(
        self,
        images: torch.Tensor,
        start_token: int,
        max_len: int = 10
    ) -> torch.Tensor:
        """
        Greedy generation (always picks most likely token).
        
        Args:
            images: Batch of chart images (B, 3, H, W)
            start_token: Starting token ID
            max_len: Maximum generation length
            
        Returns:
            Generated tokens (B, max_len) - excludes START token
        """
        self.eval()
        
        batch_size = images.size(0)
        device = images.device
        
        # Encode images
        memory = self.encoder(images)
        
        # Initialize with START token
        generated = torch.full(
            (batch_size, 1),
            start_token,
            dtype=torch.long,
            device=device
        )
        
        # Greedy generation
        for _ in range(max_len):
            logits = self.decoder(generated, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)
        
        # Return only the generated tokens, excluding the START token
        return generated[:, 1:]
