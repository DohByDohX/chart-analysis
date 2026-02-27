
import pytest
import torch
import math
from src.model.token_decoder import PositionalEncoding

class TestPositionalEncoding:
    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def max_len(self):
        return 100

    @pytest.fixture
    def dropout(self):
        return 0.1

    @pytest.fixture
    def pos_encoder(self, d_model, max_len, dropout):
        return PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

    def test_initialization(self, pos_encoder, d_model, max_len):
        """Test if PE buffer is initialized correctly."""
        # Check if buffer 'pe' exists
        assert hasattr(pos_encoder, 'pe')

        # Expected shape: (max_len, 1, d_model) based on the implementation
        # pe = pe.unsqueeze(0).transpose(0, 1) # (SeqLen, 1, Dim)
        expected_shape = (max_len, 1, d_model)
        assert pos_encoder.pe.shape == expected_shape

    def test_forward_shape(self, pos_encoder, d_model):
        """Test if forward pass preserves input shape."""
        seq_len = 10
        batch_size = 5
        x = torch.zeros(seq_len, batch_size, d_model)

        output = pos_encoder(x)
        assert output.shape == (seq_len, batch_size, d_model)

    def test_positional_values(self, d_model, max_len):
        """Test if PE values match the sinusoidal formula."""
        # Create encoder with 0 dropout to check exact values
        pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)
        pos_encoder.eval() # Ensure dropout is off

        # Get the buffer
        pe = pos_encoder.pe

        # Check specific positions
        # Formula:
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        # Check pos=0
        # sin(0) = 0, cos(0) = 1
        # pe[0, 0, 0::2] should be 0, pe[0, 0, 1::2] should be 1
        # Note: pe shape is (max_len, 1, d_model)

        # The first dimension is position.
        # The implementation uses: pe[:, 0::2] = sin(...)
        # But wait, let's look at the implementation again:
        # pe = torch.zeros(max_len, d_model)
        # pe[:, 0::2] = torch.sin(...)
        # pe[:, 1::2] = torch.cos(...)
        # pe = pe.unsqueeze(0).transpose(0, 1) -> (max_len, 1, d_model)

        # So pe[0, 0, :] corresponds to position 0.

        assert torch.allclose(pe[0, 0, 0::2], torch.zeros(d_model // 2))
        assert torch.allclose(pe[0, 0, 1::2], torch.ones(d_model // 2))

        # Check arbitrary position and dimension
        pos = 5
        i = 2 # checking dimension 2*i = 4 and 2*i+1 = 5

        # div_term calculation in code:
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # This matches 1 / 10000^(2i/d_model)

        div_term = math.exp(i * 2 * (-math.log(10000.0) / d_model))

        # In the code: position * div_term
        # position = 5

        expected_sin = math.sin(pos * div_term)
        expected_cos = math.cos(pos * div_term)

        # pe[pos, 0, 2*i]
        assert math.isclose(pe[pos, 0, 2*i].item(), expected_sin, abs_tol=1e-5)
        # pe[pos, 0, 2*i+1]
        assert math.isclose(pe[pos, 0, 2*i+1].item(), expected_cos, abs_tol=1e-5)

    def test_dropout_applied_training(self, d_model, max_len):
        """Test that dropout is applied during training."""
        pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=0.5)
        pos_encoder.train()

        seq_len = 10
        batch_size = 5
        x = torch.zeros(seq_len, batch_size, d_model)

        # With input zeros, output is dropout(pe)
        # We need to detach because dropout is stochastic
        output = pos_encoder(x)

        # Get the corresponding slice of PE
        pe_slice = pos_encoder.pe[:seq_len, :]
        # pe_slice is (seq_len, 1, d_model) -> broadcasting to (seq_len, batch_size, d_model)

        # If dropout is applied, output should not be equal to PE slice (unless very unlucky)
        # We check that at least one element is different (zeroed out or scaled)
        assert not torch.allclose(output, pe_slice)

    def test_dropout_not_applied_eval(self, d_model, max_len):
        """Test that dropout is NOT applied during evaluation."""
        pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=0.5)
        pos_encoder.eval()

        seq_len = 10
        batch_size = 5
        x = torch.zeros(seq_len, batch_size, d_model)

        output = pos_encoder(x)
        pe_slice = pos_encoder.pe[:seq_len, :]

        # Should be exactly equal (input is 0, so output = pe)
        # Broadcasting handles the batch dimension
        assert torch.allclose(output, pe_slice)

    def test_max_sequence_length(self, pos_encoder, d_model, max_len):
        """Test that it handles sequence length equal to max_len."""
        batch_size = 2
        x = torch.zeros(max_len, batch_size, d_model)
        output = pos_encoder(x)
        assert output.shape == (max_len, batch_size, d_model)

    def test_sequence_length_exceeds_max_len(self, pos_encoder, d_model, max_len):
        """Test behavior when sequence length exceeds max_len (should error)."""
        batch_size = 2
        x = torch.zeros(max_len + 1, batch_size, d_model)

        # We expect a RuntimeError or IndexError depending on implementation details
        # The implementation does: x = x + self.pe[:x.size(0), :]
        # If x.size(0) > pe.size(0), slicing might work but addition shape mismatch?
        # Actually pe[:x.size(0)] would fail if slice is larger than array? No, Python slicing clamps.
        # But if slice is smaller than x, broadcasting will fail during addition.

        # Let's verify:
        # pe has shape (max_len, 1, d_model)
        # self.pe[:max_len+1] returns self.pe (clamped to max_len)
        # So we add (max_len+1, batch, d_model) + (max_len, 1, d_model)
        # The first dimension (seq_len) mismatches.

        with pytest.raises(RuntimeError):
            pos_encoder(x)
