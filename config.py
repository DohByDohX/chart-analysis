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

# Stock symbols - Expanded to 25 tickers (S&P 500 + NASDAQ)
# Current 10: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, WMT
# Added 15 more for Phase 4 (diverse sectors for better generalization)
SAMPLE_STOCKS = [
    # Original 10 (Tech-heavy + Finance + Retail)
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "NVDA", "META", "JPM", "V", "WMT",
    
    # New 15 (S&P 500 + NASDAQ, diverse sectors)
    # Healthcare & Pharma
    "JNJ",    # Johnson & Johnson
    "UNH",    # UnitedHealth Group
    "PFE",    # Pfizer
    
    # Energy
    "XOM",    # Exxon Mobil
    "CVX",    # Chevron
    
    # Consumer & Retail
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "MCD",    # McDonald's
    
    # Finance & Insurance
    "BAC",    # Bank of America
    "GS",     # Goldman Sachs
    
    # Industrials
    "BA",     # Boeing
    "CAT",    # Caterpillar
    
    # Technology (NASDAQ)
    "INTC",   # Intel
    "NFLX",   # Netflix
    "ADBE"    # Adobe
]

# Chart settings (for Phase 1)
WINDOW_SIZE = 128  # 128-period candlestick charts (~6 months daily data)
PREDICTION_HORIZON = 5  # Predict next 5 candles
IMAGE_SIZE = (512, 512)  # 512×512 for clear visualization (4 pixels per candle)

# Vision-to-vision masking settings
CONTEXT_CANDLES = 123      # Visible candles in input image
MASKED_CANDLES = 5         # Hidden candles (WINDOW_SIZE - CONTEXT_CANDLES)
INPUT_IMAGES_DIR = DATA_DIR / "processed" / "input_images"
TARGET_IMAGES_DIR = DATA_DIR / "processed" / "target_images"

# Random sampling settings
SAMPLES_PER_STOCK = 200  # 25 stocks × 200 = 5,000 windows (Phase 4 expansion)
RANDOM_SEED = 42  # For reproducibility

# Tokenization settings
DOJI_THRESHOLD = 0.001  # Body size threshold for doji (0.1%)
PERCENTILE_LOOKBACK = 50  # Number of candles to use for percentile calculations

# Gap detection thresholds
GAP_SMALL_THRESHOLD = 0.002  # 0.2% - minimum gap to register
GAP_LARGE_THRESHOLD = 0.01   # 1.0% - threshold for large gaps

# Vocabulary: Direction (3) × Body (3) × UpperWick (4) × LowerWick (4) × Volume (3) × Gap (5) = 2,160
VOCABULARY_SIZE = 2160  # Enhanced with gap encoding

# Training hyperparameters (Phase 4.2: Extended Training)
BATCH_SIZE = 16  # Reduced for 512x512 images
LEARNING_RATE = 1e-4  # Peak learning rate after warmup
NUM_EPOCHS = 40  # Optimal for 5k dataset; early stopping likely triggers at 30-35
WARMUP_STEPS = 1500  # Warmup for ~2 epochs
LR_SCHEDULE = "cosine_restarts"  # Cosine annealing with warm restarts
COSINE_RESTART_PERIOD = 25  # Restart every 25 epochs
MIN_LR = 1e-6  # Minimum learning rate for cosine schedule

# Early stopping
EARLY_STOP_PATIENCE = 15  # Stop if no improvement for 15 epochs
EARLY_STOP_MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# Mixed precision training
USE_MIXED_PRECISION = True  # Enable automatic mixed precision (AMP)
GRADIENT_CLIP = 1.0  # Gradient clipping threshold

# Vision Encoder settings
VIT_MODEL_NAME = "google/vit-base-patch32-384"
PATCH_SIZE = 32
ENCODER_EMBED_DIM = 768  # Hidden dimension of ViT-Base
ENCODER_NUM_PATCHES = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2  # (512/32)^2 = 256
ENCODER_FREEZE_LAYERS = 10  # Freeze first 10 of 12 transformer layers

# CNN Decoder settings (vision-to-vision)
DECODER_CHANNELS = [512, 256, 128, 64, 32]  # Progressive upsampling channels

# Training settings
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOG_DIR = DATA_DIR / "logs"
GRADIENT_CLIP = 1.0

# Create directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
