"""
Configuration settings for the Vision-Trader project.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data acquisition settings
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Stock symbols
SAMPLE_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT"]

# Chart settings (for Phase 1)
WINDOW_SIZE = 100  # 100-period candlestick charts
PREDICTION_HORIZON = 10  # Predict next 10 candles
IMAGE_SIZE = (224, 224)  # Standard ViT input size

# Model settings (for future phases)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
