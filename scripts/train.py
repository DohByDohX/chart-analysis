"""
Main training script for Vision-to-Vision VisionTrader.
Orchestrates data loading, model initialization, and training loop.
"""
import sys
from pathlib import Path
import argparse
import logging
import torch
import torch.optim as optim
import json
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import ChartDataset
from src.model.vision_trader import VisionTrader
from src.training.loss import VisionTraderLoss
from src.training.trainer import Trainer
from config import (
    PROCESSED_DATA_DIR, BATCH_SIZE, LEARNING_RATE, 
    NUM_EPOCHS, WARMUP_STEPS, MIN_LR, COSINE_RESTART_PERIOD,
    VOCABULARY_SIZE  # Not used for vision-to-vision but config expects it? No, we removed it from config import in trainer.
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train VisionTrader Model")
    parser.add_argument("--run-name", type=str, default="vision_v1", help="Name of this training run")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run a short 1-epoch test")
    parser.add_argument("--no-preload", action="store_true", help="Disable RAM pre-loading")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"Starting training run: {args.run_name}")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Load Splits
    splits_path = PROCESSED_DATA_DIR / "splits.json"
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    train_ids = splits['train']
    val_ids = splits['val']
    
    if args.dry_run:
        logger.info("DRY RUN MODE: limiting data to 32 samples")
        train_ids = train_ids[:32]
        val_ids = val_ids[:16]
        args.epochs = 1
        
    # 2. Datasets & Loaders
    train_dataset = ChartDataset(
        split_name="train",
        window_ids=train_ids,
        preload_ram=not args.no_preload
    )
    val_dataset = ChartDataset(
        split_name="val",
        window_ids=val_ids,
        preload_ram=not args.no_preload
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows/multiprocessing issues often solved by 0 workers or handling main
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # 3. Model
    model = VisionTrader()
    
    # 4. Loss
    criterion = VisionTraderLoss(
        perceptual_weight=1.0,
        ssim_weight=1.0,
        masked_region_weight=10.0
    )
    
    # 5. Optimizer
    # Filter for trainable params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # 6. Scheduler (Cosine Annealing with Warm Restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=COSINE_RESTART_PERIOD,
        T_mult=1,
        eta_min=MIN_LR
    )
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        run_name=args.run_name
    )
    
    # 8. Start
    trainer.train()

if __name__ == "__main__":
    main()
