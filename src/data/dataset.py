"""
Dataset loader for Vision-to-Vision training.
Loads input/target image pairs. Supports RAM pre-loading to bypass I/O bottlenecks.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision import transforms
from tqdm import tqdm
import logging
import os
import numpy as np
from config import INPUT_IMAGES_DIR, TARGET_IMAGES_DIR, IMAGE_SIZE

logger = logging.getLogger(__name__)


class ChartDataset(Dataset):
    """
    Dataset for paired chart images (Input Masked -> Target Full).
    
    Features:
    - Loads (Input, Target) pairs based on window IDs.
    - RAM Pre-loading: Option to load all images into memory at init.
      This is CRITICAL for systems with slow I/O or NVMe bottlenecks.
    - Transforms: Converts to Tensor and normalizes to [0, 1].
    """
    
    def __init__(
        self,
        split_name: str,
        window_ids: list,
        preload_ram: bool = True
    ):
        """
        Args:
            split_name: 'train', 'val', or 'test' (for logging)
            window_ids: List of window IDs to include in this dataset
            preload_ram: If True, loads all images into RAM at initialization
        """
        self.split_name = split_name
        self.window_ids = window_ids
        self.preload_ram = preload_ram
        self.data_cache = []
        
        # Input transform: ToTensor (0-1) + Normalize (-1 to 1 for ViT)
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Target transform: Just ToTensor (0-1) for Sigmoid output
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if self.preload_ram:
            self._preload_data()
            
    def _preload_data(self):
        """Load all images into RAM as uint8 tensors."""
        logger.info(f"[{self.split_name}] Pre-loading {len(self.window_ids)} pairs into RAM...")
        
        # Use tqdm for progress bar
        for wid in tqdm(self.window_ids, desc=f"Loading {self.split_name}", unit="img"):
            input_path = INPUT_IMAGES_DIR / f"input_{wid}.png"
            target_path = TARGET_IMAGES_DIR / f"target_{wid}.png"
            
            try:
                # Load images
                input_img = Image.open(input_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
                
                # Convert to Tensor (uint8) (C, H, W)
                # PIL -> ByteTensor [0, 255]
                # transforms.PILToTensor() preserves uint8
                input_tensor = torch.from_numpy(np.array(input_img)).permute(2, 0, 1)
                target_tensor = torch.from_numpy(np.array(target_img)).permute(2, 0, 1)
                
                self.data_cache.append((input_tensor, target_tensor))
                
            except Exception as e:
                logger.warning(f"Failed to load window {wid}: {e}")

        logger.info(f"[{self.split_name}] Loaded {len(self.data_cache)} pairs.")

    def __len__(self):
        # If preloaded, use cache length. Otherwise use window_ids length.
        if self.preload_ram:
            return len(self.data_cache)
        return len(self.window_ids)

    def __getitem__(self, idx):
        if self.preload_ram:
            input_tensor_u8, target_tensor_u8 = self.data_cache[idx]
            # Convert uint8 [0, 255] -> float [0.0, 1.0]
            input_tensor = input_tensor_u8.float() / 255.0
            target_tensor = target_tensor_u8.float() / 255.0
            
            # Apply normalization to input only (0..1 -> -1..1)
            input_tensor = (input_tensor - 0.5) / 0.5
        else:
            # Fallback for lazy loading
            wid = self.window_ids[idx]
            input_path = INPUT_IMAGES_DIR / f"input_{wid}.png"
            target_path = TARGET_IMAGES_DIR / f"target_{wid}.png"
            
            with Image.open(input_path) as img:
                input_img = img.convert("RGB")
            with Image.open(target_path) as img:
                target_img = img.convert("RGB")
            
            input_tensor = self.input_transform(input_img)
            target_tensor = self.target_transform(target_img)
        
        return input_tensor, target_tensor
