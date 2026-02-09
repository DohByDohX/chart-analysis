import torch
import torch.nn as nn
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1) # (SeqLen, 1, Dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [SeqLen, Batch, Dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

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
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output Head
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        logger.info(f"Token Decoder initialized. Layers: {num_layers}, Heads: {num_heads}")

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

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
        # PyTorch Transformer expects (SeqLen, Batch, Dim) by default
        # But we can use batch_first=False which is default.
        # So we transpose inputs: (Batch, Seq) -> (Seq, Batch)
        
        tgt = self.embedding(tgt_tokens) * math.sqrt(self.embed_dim)
        tgt = tgt.transpose(0, 1) # (TgtSeqLen, Batch, Dim)
        tgt = self.pos_encoder(tgt)
        
        memory = memory.transpose(0, 1) # (SrcSeqLen, Batch, Dim)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            device = tgt.device
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(device)
            
        # Transformer Decoder pass
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # Output Head
        output = self.output_head(output) # (TgtSeqLen, Batch, VocabSize)
        
        # Transpose back to (Batch, TgtSeqLen, VocabSize)
        return output.transpose(0, 1)
