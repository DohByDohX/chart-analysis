import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
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
    MAX_TGT_SEQ_LEN, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    GRADIENT_CLIP, CHECKPOINT_DIR, LOG_DIR, DATA_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, target_tokens) in enumerate(pbar):
        images = images.to(device)
        target_tokens = target_tokens.to(device)
        
        # Forward pass
        logits = model(images, target_tokens)  # (B, SeqLen, VocabSize)
        
        # Calculate loss (flatten for CrossEntropyLoss)
        # logits: (B, SeqLen, VocabSize) -> (B*SeqLen, VocabSize)
        # target_tokens: (B, SeqLen) -> (B*SeqLen)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    logger.info("=" * 70)
    logger.info("VisionTrader Training")
    logger.info("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = CandleTokenizer()
    tokenizer_path = DATA_DIR / "processed" / "tokenizer" / "vocabulary.json"
    if tokenizer_path.exists():
        tokenizer.load_vocabulary(str(tokenizer_path))
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
    else:
        logger.warning(f"Tokenizer not found at {tokenizer_path}, using default")
    
    # Create dataset
    logger.info("Loading dataset...")
    windows_dir = DATA_DIR / "processed" / "windows"
    images_dir = DATA_DIR / "processed" / "images"
    
    dataset = ChartDataset(
        windows_dir=windows_dir,
        images_dir=images_dir,
        tokenizer=tokenizer
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Number of batches: {len(dataloader)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = VisionTrader(
        vit_model_name=VIT_MODEL_NAME,
        vocab_size=VOCABULARY_SIZE,
        embed_dim=ENCODER_EMBED_DIM,
        num_heads=DECODER_NUM_HEADS,
        num_layers=DECODER_NUM_LAYERS,
        dropout=DECODER_DROPOUT,
        max_seq_len=MAX_TGT_SEQ_LEN
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Optimizer: Adam, LR: {LEARNING_RATE}")
    logger.info(f"Loss: CrossEntropyLoss")
    
    # Training loop
    logger.info("\nStarting training...\n")
    best_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("\nTraining completed!")
    logger.info(f"Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
