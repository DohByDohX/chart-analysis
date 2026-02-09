import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartDataset(Dataset):
    """
    PyTorch Dataset for chart images and tokenized sequences.
    """
    
    def __init__(
        self,
        windows_dir: Path,
        images_dir: Path,
        tokenizer,
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            windows_dir: Directory containing window JSON files
            images_dir: Directory containing chart images
            tokenizer: CandleTokenizer instance
            transform: Optional image transforms
        """
        self.windows_dir = Path(windows_dir)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load all window files
        self.window_files = sorted(list(self.windows_dir.glob("window_*.json")))
        
        if len(self.window_files) == 0:
            raise ValueError(f"No window files found in {windows_dir}")
        
        logger.info(f"Loaded {len(self.window_files)} windows from {windows_dir}")
    
    def __len__(self):
        return len(self.window_files)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            image: Tensor (3, H, W)
            target_tokens: Tensor (SeqLen,)
        """
        # Load window data
        window_file = self.window_files[idx]
        with open(window_file, 'r') as f:
            window_data = json.load(f)
        
        # Extract window ID from filename (e.g., "window_00001.json" -> "00001")
        window_id = window_file.stem.split('_')[1]
        
        # Load corresponding image
        image_path = self.images_dir / f"chart_{window_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image)
        
        # Tokenize target window
        target_window = window_data['target_window']
        target_tokens = self.tokenizer.tokenize_window(target_window)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)
        
        return image, target_tokens
    
    def get_window_data(self, idx):
        """Get the raw window data for inspection."""
        window_file = self.window_files[idx]
        with open(window_file, 'r') as f:
            return json.load(f)
