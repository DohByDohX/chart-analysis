"""
Trainer class for Vision-to-Vision training.
Handles training loop, validation, checkpointing, and logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image

# Project imports
from config import (
    CHECKPOINT_DIR, LOG_DIR, 
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
    GRADIENT_CLIP, USE_MIXED_PRECISION
)

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        device: torch.device = None,
        num_epochs: int = 20,
        save_every: int = 1,
        run_name: str = "default",
        accumulation_steps: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.run_name = run_name
        self.accumulation_steps = accumulation_steps
        
        # Setup directories
        self.checkpoint_dir = CHECKPOINT_DIR / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = LOG_DIR / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # AMP Scaler
        self.scaler = GradScaler(enabled=USE_MIXED_PRECISION)
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        self.model.to(self.device)
        self.criterion.to(self.device)
        logger.info(f"Trainer initialized on {self.device}. Run: {run_name}")

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train Epoch
            train_loss = self.train_epoch(epoch)
            
            # Validation Epoch
            val_loss = self.validate(epoch)
            
            # Learning Rate Scheduler Step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch+1} LR: {current_lr:.6f}")
                
            # Checkpoint (Best & Regular)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
                
            # Early Stopping
            if self.epochs_no_improve >= EARLY_STOP_PATIENCE:
                logger.info(f"Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement.")
                break
                
            # Log simple progress
            logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=True)
        
        for batch_idx, (inputs, targets) in enumerate(loop):
            try:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero gradients at start of accumulation block (handled by step logic, but good for safety)
                # self.optimizer.zero_grad() -> Moved to step logic
                
                # Forward Pass with AMP
                with autocast(enabled=USE_MIXED_PRECISION):
                    outputs = self.model(inputs)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['loss']
                
                # Backward Pass with Scaler
                self.scaler.scale(loss).backward()
                
                # Gradient Accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Clip gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                    
                    # Optimizer Step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Update progress bar
                loop.set_postfix(loss=loss.item())
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}", exc_info=True)
                raise e
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch: int) -> float:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]", leave=True)
            
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast(enabled=USE_MIXED_PRECISION):
                    outputs = self.model(inputs)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['loss']
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
                # Save first batch images for visualization
                if batch_idx == 0:
                    self.save_visualization(inputs, outputs, targets, epoch)
                    
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save training checkpoint."""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': {
                'run_name': self.run_name,
                'num_epochs': self.num_epochs
            }
        }
        
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
        path = self.checkpoint_dir / filename
        torch.save(state, path)
        
    def save_visualization(self, inputs, outputs, targets, epoch):
        """Save a grid of input/pred/target images."""
        # limit to 4 samples
        n = min(inputs.size(0), 4)
        
        # Helper to convert tensor to numpy image (H, W, C) range [0, 255]
        def to_img(t, denorm=False):
            if denorm:
                t = (t * 0.5) + 0.5
            t = t.detach().cpu().numpy().transpose(1, 2, 0)
            t = np.clip(t, 0, 1)
            return (t * 255).astype(np.uint8)
            
        combined_rows = []
        for i in range(n):
            img_in = to_img(inputs[i], denorm=True)
            img_pred = to_img(outputs[i])
            img_tgt = to_img(targets[i])
            
            # Concatenate horizontally: Input | Prediction | Target
            # Add a small white separator
            sep = np.ones((512, 10, 3), dtype=np.uint8) * 255
            row = np.concatenate([img_in, sep, img_pred, sep, img_tgt], axis=1)
            combined_rows.append(row)
            
        # Concatenate vertically
        if combined_rows:
            # Add horizontal separator
            h_sep = np.ones((10, combined_rows[0].shape[1], 3), dtype=np.uint8) * 255
            final_img = combined_rows[0]
            for row in combined_rows[1:]:
                final_img = np.concatenate([final_img, h_sep, row], axis=0)
                
            save_path = self.log_dir / f"viz_epoch_{epoch+1}.png"
            Image.fromarray(final_img).save(save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Resuming from epoch {self.start_epoch} with best_val_loss: {self.best_val_loss:.4f}")
