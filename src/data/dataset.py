"""
Dataset loader for Vision-to-Vision training.
Loads input/target image pairs. Supports RAM pre-loading to bypass I/O bottlenecks.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import logging
import os
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
        
        # Standard transform: ToTensor (0-255 -> 0.0-1.0)
        # We perform resizing just in case, though images should already be 512x512
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if self.preload_ram:
            self._preload_data()
            
    def _preload_data(self):
        """Load all images into RAM."""
        logger.info(f"[{self.split_name}] Pre-loading {len(self.window_ids)} pairs into RAM...")
        
        # Use tqdm for progress bar
        for wid in tqdm(self.window_ids, desc=f"Loading {self.split_name}", unit="img"):
            input_path = INPUT_IMAGES_DIR / f"input_{wid}.png"
            target_path = TARGET_IMAGES_DIR / f"target_{wid}.png"
            
            try:
                # Load images
                # Convert to RGB to ensure 3 channels
                input_img = Image.open(input_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
                
                # Apply transforms immediately to save RAM (tensors are packed)
                # Note: Keeping as PIL might be more compact, but converting to Tensor
                # here saves CPU time during training.
                # 512x512x3 float32 tensor = 3 MB. 5000 images = 15 GB.
                # If 15GB is too much, we can transform on the fly.
                # The user has 32GB RAM. 15GB is significant but might fit.
                # Let's try transforming on the fly to be safer with RAM,
                # storing PIL images usually takes less space (though they are decompressed).
                # Actually, raw pixel data in PIL is also bytes. 512*512*3 bytes = 0.75 MB.
                # 0.75 MB * 5000 = 3.75 GB. This is MUUUCH better.
                # So we store PIL images in RAM, and transform in __getitem__.
                
                # Force loading pixel data
                input_img.load()
                target_img.load()
                
                self.data_cache.append((input_img, target_img))
                
            except Exception as e:
                logger.warning(f"Failed to load window {wid}: {e}")
                # We might have a mismatch in lengths if we skip, but for now just warn
                # For strict correctness, we should remove this ID from the list,
                # but simplistic approach for now.
                
        logger.info(f"[{self.split_name}] Loaded {len(self.data_cache)} pairs.")

    def __len__(self):
        # If preloaded, use cache length. Otherwise use window_ids length.
        if self.preload_ram:
            return len(self.data_cache)
        return len(self.window_ids)

    def __getitem__(self, idx):
        if self.preload_ram:
            input_img, target_img = self.data_cache[idx]
        else:
            wid = self.window_ids[idx]
            input_path = INPUT_IMAGES_DIR / f"input_{wid}.png"
            target_path = TARGET_IMAGES_DIR / f"target_{wid}.png"
            
            input_img = Image.open(input_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
        
        # Transform to tensor
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)
        
        return input_tensor, target_tensor
