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
IMAGE_DATA_DIR = DATA_DIR / "images" # Added IMAGE_DATA_DIR
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGES_DIR, MODELS_DIR, IMAGE_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data acquisition settings
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Stock symbols
SAMPLE_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT"]

# Chart settings (for Phase 1)
WINDOW_SIZE = 128  # 128-period candlestick charts (~6 months daily data)
PREDICTION_HORIZON = 5  # Predict next 5 candles
IMAGE_SIZE = (512, 512)  # 512×512 for clear visualization (4 pixels per candle)

# Random sampling settings
SAMPLES_PER_STOCK = 100  # Number of random windows to generate per stock
RANDOM_SEED = 42  # For reproducibility

# Tokenization settings
DOJI_THRESHOLD = 0.001  # Body size threshold for doji (0.1%)
PERCENTILE_LOOKBACK = 50  # Number of candles to use for percentile calculations
VOCABULARY_SIZE = 432  # Total number of possible tokens (3*3*4*4*3)

# Model settings (Phase 2)
BATCH_SIZE = 16  # Reduced for 512x512 images
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Vision Encoder settings
VIT_MODEL_NAME = "google/vit-base-patch32-384"
PATCH_SIZE = 32
ENCODER_EMBED_DIM = 768  # Hidden dimension of ViT-Base
SEQUENCE_LENGTH = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2 + 1  # (512/32)^2 + 1 = 257

# Decoder settings
DECODER_NUM_LAYERS = 6
DECODER_NUM_HEADS = 8
DECODER_DROPOUT = 0.1
MAX_TGT_SEQ_LEN = 16  # 5-10 candles + buffer
