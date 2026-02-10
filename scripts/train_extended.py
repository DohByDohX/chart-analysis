"""
Extended training script for VisionTrader - Phase 4.2

Improvements:
- Early stopping with patience
- Cosine annealing with warm restarts
- Mixed precision training (AMP)
- Enhanced checkpointing
- Training/validation split

Per OSVariables.md:
- Training is GPU-bound (not I/O intensive)
- Memory usage: <2 GB per batch (within limits)
- Can run in parallel with dataset generation
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW  # AdamW with weight decay
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler  # Mixed precision
from tqdm import tqdm
import logging
import numpy as np

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
    GRADIENT_CLIP, CHECKPOINT_DIR, LOG_DIR, DATA_DIR, START_TOKEN,
    COSINE_RESTART_PERIOD, MIN_LR, EARLY_STOP_PATIENCE, 
    EARLY_STOP_MIN_DELTA, USE_MIXED_PRECISION
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'training_extended.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp):
    """Train for one epoch with mixed precision support."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for images, target_tokens in pbar:
        images = images.to(device)
        target_tokens = target_tokens.to(device)
        
        # Prepend START token
        batch_size = target_tokens.size(0)
        start_tokens = torch.full((batch_size, 1), START_TOKEN, dtype=torch.long, device=device)
        decoder_input = torch.cat([start_tokens, target_tokens[:, :-1]], dim=1)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if use_amp:
            with autocast():
                logits = model(images, decoder_input)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1)
                )
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, decoder_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, use_amp):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, target_tokens in dataloader:
            images = images.to(device)
            target_tokens = target_tokens.to(device)
            
            batch_size = target_tokens.size(0)
            start_tokens = torch.full((batch_size, 1), START_TOKEN, dtype=torch.long, device=device)
            decoder_input = torch.cat([start_tokens, target_tokens[:, :-1]], dim=1)
            
            if use_amp:
                with autocast():
                    logits = model(images, decoder_input)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        target_tokens.reshape(-1)
                    )
            else:
                logits = model(images, decoder_input)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1)
                )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    logger.info("=" * 70)
    logger.info("VisionTrader Extended Training (Phase 4.2)")
    logger.info("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    use_amp = USE_MIXED_PRECISION and device.type == 'cuda'
    if use_amp:
        logger.info("Mixed precision training ENABLED")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = CandleTokenizer()
    tokenizer_path = DATA_DIR / "processed" / "tokenizer" / "vocabulary.json"
    if token izer_path.exists():
        tokenizer.load_vocabulary(str(tokenizer_path))
    
    # Create dataset
    logger.info("Loading dataset...")
    windows_dir = DATA_DIR / "processed" / "windows"
    images_dir = DATA_DIR / "processed" / "images"
    
    full_dataset = ChartDataset(
        windows_dir=windows_dir,
        images_dir=images_dir,
        tokenizer=tokenizer
    )
    
    # Train/validation split (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Total dataset: {len(full_dataset)}")
    logger.info(f"Training samples: {train_size}")
    logger.info(f"Validation samples: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=COSINE_RESTART_PERIOD,
        T_mult=1,
        eta_min=MIN_LR
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay=0.01)")
    logger.info(f"LR Scheduler: CosineAnnealingWarmRestarts (T_0={COSINE_RESTART_PERIOD}, eta_min={MIN_LR})")
    logger.info(f"Early stopping: patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    
    # Training loop with early stopping
    logger.info(f"\nStarting training for up to {NUM_EPOCHS} epochs...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, use_amp)
        
        # Step scheduler
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = CHECKPOINT_DIR / "best_model_extended.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
        
        # Early stopping check
        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"\nEarly stopping triggered at epoch {epoch}")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            break
        
        # Regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}_extended.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: epoch {epoch}")
        
        # Clear cache every 5 epochs (per OSVariables.md)
        if epoch % 5 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 70)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
