import torch
import torch.nn as nn
import math
import logging
from .utils import PositionalEncoding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for predicting candle tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 16
    ):
        """
        Initialize the Token Decoder.
        
        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Embedding dimension (must match Encoder output)
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional Encoding
        # ⚡ Bolt: Enable batch_first=True to avoid multiple transpositions
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout, batch_first=True)
        
        # Transformer Decoder
        # ⚡ Bolt: Use batch_first=True to natively accept (Batch, SeqLen, Dim)
        # avoiding costly transpose operations that create non-contiguous tensors.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output Head
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Cache for causal masks to prevent reallocation and cross-device sync
        self._mask_cache = {}

        logger.info(f"Token Decoder initialized. Layers: {num_layers}, Heads: {num_heads}")

    def generate_square_subsequent_mask(self, sz: int, device: torch.device = None) -> torch.Tensor:
        """Generate causal mask to prevent attending to future tokens."""
        # ⚡ Bolt: Cache masks by size and device to avoid O(N^2) memory reallocation
        # and cross-device synchronization latency on every forward pass.
        cache_key = (sz, device)
        if cache_key not in self._mask_cache:
            # ⚡ Bolt: Use PyTorch's optimized implementation directly on target device
            self._mask_cache[cache_key] = torch.nn.Transformer.generate_square_subsequent_mask(
                sz, device=device
            )
        return self._mask_cache[cache_key]

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt_tokens: Target token sequence (Batch, TgtSeqLen)
            memory: Encoder output (Batch, SrcSeqLen, EmbedDim)
            tgt_mask: Causal mask (TgtSeqLen, TgtSeqLen)
            
        Returns:
            Logits (Batch, TgtSeqLen, VocabSize)
        """
        # ⚡ Bolt: With batch_first=True, we process directly in (Batch, SeqLen, Dim) format.
        # This removes three transpose() operations per forward pass, keeping memory contiguous.
        
        # ⚡ Bolt Optimization: Use in-place multiplication (.mul_) on the newly created
        # embedding tensor to avoid allocating a second temporary tensor for the result.
        tgt = self.embedding(tgt_tokens).mul_(math.sqrt(self.embed_dim))
        tgt = self.pos_encoder(tgt) # (Batch, TgtSeqLen, Dim)
        
        # memory is already (Batch, SrcSeqLen, Dim) from VisionEncoder
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            # ⚡ Bolt: Pass target device to mask generator to avoid .to() latency
            # tgt is (Batch, TgtSeqLen, Dim), so size(1) is TgtSeqLen
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
            
        # Transformer Decoder pass
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        ) # (Batch, TgtSeqLen, Dim)
        
        # Output Head
        output = self.output_head(output) # (Batch, TgtSeqLen, VocabSize)
        
        return output
