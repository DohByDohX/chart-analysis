"""
Sanity check: Analyze model architecture and layer outputs.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.transformer import VisionTrader
from src.training.dataset import ChartDataset
from src.data.tokenizer import CandleTokenizer
from config import (
    VIT_MODEL_NAME, VOCABULARY_SIZE, ENCODER_EMBED_DIM,
    DECODER_NUM_LAYERS, DECODER_NUM_HEADS, DECODER_DROPOUT,
    MAX_TGT_SEQ_LEN, DATA_DIR, CHECKPOINT_DIR, START_TOKEN
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gradients(model):
    """Check if gradients are flowing properly."""
    logger.info("\n--- Gradient Flow Check ---")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            logger.info(f"{name:60s} | grad_norm: {grad_norm:.6f}")
        else:
            logger.info(f"{name:60s} | NO GRADIENT")

def main():
    logger.info("=" * 70)
    logger.info("Architecture Sanity Check")
    logger.info("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load tokenizer and dataset
    tokenizer = CandleTokenizer()
    tokenizer_path = DATA_DIR / "processed" / "tokenizer" / "vocabulary.json"
    tokenizer.load_vocabulary(str(tokenizer_path))
    
    windows_dir = DATA_DIR / "processed" / "windows"
    images_dir = DATA_DIR / "processed" / "images"
    dataset = ChartDataset(windows_dir=windows_dir, images_dir=images_dir, tokenizer=tokenizer)
    
    # Initialize model
    model = VisionTrader(
        vit_model_name=VIT_MODEL_NAME,
        vocab_size=VOCABULARY_SIZE,
        embed_dim=ENCODER_EMBED_DIM,
        num_heads=DECODER_NUM_HEADS,
        num_layers=DECODER_NUM_LAYERS,
        dropout=DECODER_DROPOUT,
        max_seq_len=MAX_TGT_SEQ_LEN
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get a sample
    image, target_tokens = dataset[0]
    image_batch = image.unsqueeze(0).to(device)
    target_batch = target_tokens.unsqueeze(0).to(device)
    
    # Create decoder input
    start_tokens = torch.full((1, 1), START_TOKEN, dtype=torch.long, device=device)
    decoder_input = torch.cat([start_tokens, target_batch[:, :-1]], dim=1)
    
    logger.info("\n--- Input Shapes ---")
    logger.info(f"Image: {image_batch.shape}")
    logger.info(f"Decoder input: {decoder_input.shape}")
    logger.info(f"Target: {target_batch.shape}")
    
    # Test 1: Check encoder output
    logger.info("\n=== TEST 1: Encoder Output ===")
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(image_batch)
    logger.info(f"Shape: {encoder_output.shape}")
    logger.info(f"Mean: {encoder_output.mean():.4f}, Std: {encoder_output.std():.4f}")
    logger.info(f"Min: {encoder_output.min():.4f}, Max: {encoder_output.max():.4f}")
    
    # Check if encoder output varies across sequence
    position_variance = encoder_output[0].var(dim=0).mean().item()
    logger.info(f"Variance across positions: {position_variance:.4f}")
    if position_variance < 0.01:
        logger.warning("⚠️  LOW VARIANCE - Encoder may not be encoding spatial information!")
    
    # Test 2: Check decoder embeddings
    logger.info("\n=== TEST 2: Decoder Embeddings ===")
    with torch.no_grad():
        embeddings = model.decoder.embedding(decoder_input)
    logger.info(f"Shape: {embeddings.shape}")
    logger.info(f"Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
    
    # Check if different tokens have different embeddings
    token_0_emb = model.decoder.embedding(torch.tensor([[0]], device=device))
    token_100_emb = model.decoder.embedding(torch.tensor([[100]], device=device))
    token_200_emb = model.decoder.embedding(torch.tensor([[200]], device=device))
    
    dist_0_100 = torch.dist(token_0_emb, token_100_emb).item()
    dist_0_200 = torch.dist(token_0_emb, token_200_emb).item()
    dist_100_200 = torch.dist(token_100_emb, token_200_emb).item()
    
    logger.info(f"Distance token 0-100: {dist_0_100:.4f}")
    logger.info(f"Distance token 0-200: {dist_0_200:.4f}")
    logger.info(f"Distance token 100-200: {dist_100_200:.4f}")
    
    if dist_0_100 < 1.0 or dist_0_200 < 1.0:
        logger.warning("⚠️  Embeddings are too similar!")
    
    # Test 3: Check full forward pass
    logger.info("\n=== TEST 3: Full Forward Pass ===")
    with torch.no_grad():
        logits = model(image_batch, decoder_input)
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    
    # Check if logits vary across positions
    position_logit_variance = logits[0].var(dim=0).mean().item()
    logger.info(f"Variance across positions: {position_logit_variance:.4f}")
    if position_logit_variance < 0.1:
        logger.warning("⚠️  LOW VARIANCE - Model may be predicting same thing everywhere!")
    
    # Test 4: Check if model responds to different inputs
    logger.info("\n=== TEST 4: Input Sensitivity ===")
    
    # Try 3 different images
    samples = [0, 50, 100]
    predictions = []
    
    for idx in samples:
        img, _ = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        test_input = torch.full((1, 1), START_TOKEN, dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model(img_batch, test_input)
            pred = logits[0, 0].argmax().item()
            predictions.append(pred)
        
        logger.info(f"Sample {idx}: predicts token {pred}")
    
    if len(set(predictions)) == 1:
        logger.error("❌ CRITICAL: Model predicts SAME token for different images!")
    else:
        logger.info(f"✓ Model produces {len(set(predictions))} different predictions")
    
    # Test 5: Gradient flow check (requires training mode)
    logger.info("\n=== TEST 5: Gradient Flow ===")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    logits = model(image_batch, decoder_input)
    loss = criterion(logits.reshape(-1, logits.size(-1)), target_batch.reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    encoder_has_grad = False
    decoder_has_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'encoder' in name:
                encoder_has_grad = True
            if 'decoder' in name:
                decoder_has_grad = True
    
    logger.info(f"Encoder has gradients: {encoder_has_grad}")
    logger.info(f"Decoder has gradients: {decoder_has_grad}")
    
    if not encoder_has_grad:
        logger.error("❌ CRITICAL: Encoder not receiving gradients!")
    if not decoder_has_grad:
        logger.error("❌ CRITICAL: Decoder not receiving gradients!")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Gradient flow: {'✓ OK' if encoder_has_grad and decoder_has_grad else '❌ ISSUE'}")
    logger.info(f"Input sensitivity: {'✓ OK' if len(set(predictions)) > 1 else '❌ ISSUE'}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
