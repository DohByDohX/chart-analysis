"""
Chart renderer for creating monochrome candlestick chart images.
Renders clean 512×512 charts with zoom-to-fit normalization.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartRenderer:
    """
    Renders candlestick charts as color images for Vision Transformer training.
    
    Features:
    - Clean, minimal design (no grid, labels, or UI elements)
    - Zoom-to-fit normalization (min-max per window)
    - 512×512 output with 3-channel RGB
    - 4 pixels per candle for clear visualization
    - Vibrant green/red color scheme for better distinction
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (512, 512),
        dpi: int = 100
    ):
        """
        Initialize the chart renderer.
        
        Args:
            output_size: Output image size in pixels (width, height)
            dpi: Dots per inch for rendering
        """
        self.output_size = output_size
        self.dpi = dpi
        self.fig_size = (output_size[0] / dpi, output_size[1] / dpi)
        
        logger.info(f"Initialized ChartRenderer: {output_size[0]}×{output_size[1]} @ {dpi} DPI")
    
    def apply_zoom_to_fit(self, window_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply zoom-to-fit normalization to price data.
        
        Normalizes all prices to [0, 1] range based on window's min/max.
        
        Args:
            window_df: DataFrame with OHLCV data
            
        Returns:
            Normalized DataFrame
        """
        normalized = window_df.copy()
        
        # Get price range for the window
        price_min = window_df['Low'].min()
        price_max = window_df['High'].max()
        price_range = price_max - price_min
        
        if price_range == 0:
            price_range = 1  # Avoid division by zero
        
        # Normalize OHLC to [0, 1]
        for col in ['Open', 'High', 'Low', 'Close']:
            normalized[col] = (window_df[col] - price_min) / price_range
        
        # Normalize volume separately to [0, 1]
        vol_min = window_df['Volume'].min()
        vol_max = window_df['Volume'].max()
        vol_range = vol_max - vol_min
        
        if vol_range == 0:
            vol_range = 1
        
        normalized['Volume'] = (window_df['Volume'] - vol_min) / vol_range
        
        return normalized
    
    def render_window(
        self,
        window_df: pd.DataFrame,
        apply_normalization: bool = True,
        total_candles: Optional[int] = None
    ) -> np.ndarray:
        """
        Render a window as a monochrome candlestick chart image.
        
        Args:
            window_df: DataFrame with OHLCV data
            apply_normalization: Whether to apply zoom-to-fit normalization
            total_candles: If set, fix x-axis to this width regardless of
                DataFrame length. Used for rendering partial charts (e.g.,
                123 candles in a 128-candle layout with empty space on right).
            
        Returns:
            NumPy array of shape (height, width, 3) with values in [0, 1]
        """
        # Apply zoom-to-fit normalization
        if apply_normalization:
            data = self.apply_zoom_to_fit(window_df)
        else:
            data = window_df.copy()
        
        # Create figure with no margins
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi, facecolor='black')
        
        # Create axes for candlesticks (top 80%) and volume (bottom 20%)
        ax_candles = fig.add_axes([0, 0.2, 1, 0.8], facecolor='black')
        ax_volume = fig.add_axes([0, 0, 1, 0.2], facecolor='black')
        
        # Render candlesticks
        self._render_candlesticks(ax_candles, data)
        
        # Render volume
        self._render_volume(ax_volume, data)
        
        # Remove all axes elements
        # Use total_candles for x-axis width if specified, otherwise fit to data
        xlim_end = (total_candles - 0.5) if total_candles else (len(data) - 0.5)
        for ax in [ax_candles, ax_volume]:
            ax.set_xlim(-0.5, xlim_end)
            ax.axis('off')
        
        ax_candles.set_ylim(-0.05, 1.05)
        ax_volume.set_ylim(0, 1.05)
        
        # Convert to numpy array
        fig.canvas.draw()
        # Use buffer_rgba() for newer matplotlib versions
        buf = fig.canvas.buffer_rgba()
        image_array = np.asarray(buf).copy()  # Make a copy before closing
        
        plt.close(fig)
        
        # Convert RGBA to RGB and normalize to [0, 1]
        rgb_image = image_array[:, :, :3].astype(np.float32) / 255.0
        
        # Clear any remaining matplotlib state
        plt.clf()
        
        return rgb_image
    
    def _render_candlesticks(self, ax, data: pd.DataFrame):
        """Render candlestick bodies and wicks."""
        for i, (idx, row) in enumerate(data.iterrows()):
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Determine candle color (bullish=green, bearish=red)
            is_bullish = close_price >= open_price
            body_color = '#089981' if is_bullish else '#F23645'  # TradingView colors
            wick_color = body_color  # Match wick to body color
            
            # Draw wicks (high-low line)
            ax.plot([i, i], [low_price, high_price], 
                   color=wick_color, linewidth=1.0, solid_capstyle='round')
            
            # Draw body
            body_bottom = min(open_price, close_price)
            body_height = abs(close_price - open_price)
            
            if body_height < 0.001:  # Doji
                ax.plot([i - 0.375, i + 0.375], [open_price, open_price],
                       color=wick_color, linewidth=1.0)
            else:
                body = patches.Rectangle(
                    (i - 0.375, body_bottom),
                    0.75,
                    body_height,
                    linewidth=0,
                    edgecolor=None,
                    facecolor=body_color
                )
                ax.add_patch(body)
    
    def _render_volume(self, ax, data: pd.DataFrame):
        """Render volume bars."""
        for i, (idx, row) in enumerate(data.iterrows()):
            volume = row['Volume']
            
            # Determine color based on price direction
            is_bullish = row['Close'] >= row['Open']
            bar_color = '#089981' if is_bullish else '#F23645'  # Match candle colors
            
            # Draw volume bar
            bar = patches.Rectangle(
                (i - 0.375, 0),
                0.75,
                volume,
                linewidth=0,
                edgecolor=None,
                facecolor=bar_color
            )
            ax.add_patch(bar)
    
    def to_grayscale_3channel(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale image to 3-channel (stacking trick).
        
        Args:
            image: Grayscale image of shape (H, W)
            
        Returns:
            3-channel image of shape (H, W, 3)
        """
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            return np.repeat(image, 3, axis=-1)
        else:
            return image
    
    def save_image(
        self,
        image: np.ndarray,
        filepath: Union[str, Path],
        format: str = 'png'
    ):
        """
        Save rendered image to file.
        
        Args:
            image: Image array (values in [0, 1])
            filepath: Output file path
            format: Image format (png, jpg, etc.)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Save using PIL
        pil_image = Image.fromarray(image_uint8)
        pil_image.save(filepath, format=format.upper())
        
        logger.info(f"Saved image to {filepath}")
    
    def render_batch(
        self,
        windows: list,
        output_dir: Union[str, Path],
        save_images: bool = True,
        return_images: bool = False
    ) -> list:
        """
        Render multiple windows as images.
        
        Args:
            windows: List of window dictionaries (from WindowGenerator)
            output_dir: Directory to save images
            save_images: Whether to save images to disk
            return_images: Whether to return the rendered images in memory (WARNING: High memory usage for large batches)
            
        Returns:
            List of rendered images (numpy arrays) if return_images is True, else empty list
        """
        output_dir = Path(output_dir)
        rendered_images = []
        import gc
        
        for i, window in enumerate(windows):
            # Render input window
            image = self.render_window(window['input'])
            
            if return_images:
                rendered_images.append({
                    'symbol': window['symbol'],
                    'window_index': window['window_index'],
                    'image': image,
                    'start_date': window['start_date'],
                    'end_date': window['end_date']
                })
            
            # Save if requested
            if save_images:
                filename = f"{window['symbol']}_{window['window_index']:03d}.png"
                self.save_image(image, output_dir / filename)
            
            # Explicitly delete image reference to free memory
            if not return_images:
                del image
            
            # Periodically force garbage collection (every 100 images or if logic suggests)
            # Matplotlib can be leaky, so being aggressive here for safety
            if i % 10 == 0:
                gc.collect()
        
        if return_images:
            logger.info(f"Rendered {len(rendered_images)} images")
        else:
            logger.info(f"Rendered {len(windows)} images to disk")
            
        return rendered_images
