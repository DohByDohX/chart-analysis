"""
Future candle renderer for visualizing model predictions.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FutureRenderer:
    """
    Renders predicted candlestick charts with distinctive styling.
    
    Features:
    - Dashed borders for predicted candles
    - Semi-transparent fills
    - Side-by-side and overlay comparison modes
    - Clear visual distinction from actual data
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (1024, 512),
        dpi: int = 100
    ):
        """
        Initialize the future renderer.
        
        Args:
            output_size: Output image size in pixels (width, height)
            dpi: Dots per inch for rendering
        """
        self.output_size = output_size
        self.dpi = dpi
        self.fig_size = (output_size[0] / dpi, output_size[1] / dpi)
        
        # Color scheme
        self.actual_colors = {
            'bullish': '#26A69A',  # Teal green
            'bearish': '#EF5350'   # Red
        }
        self.predicted_colors = {
            'bullish': '#42A5F5',  # Blue
            'bearish': '#FF9800'   # Orange
        }
        
        logger.info(f"Initialized FutureRenderer: {output_size[0]}×{output_size[1]} @ {dpi} DPI")
    
    def render_candlesticks(
        self,
        ax,
        data: pd.DataFrame,
        x_offset: int = 0,
        is_predicted: bool = False,
        show_volume: bool = True
    ):
        """
        Render candlesticks on the given axis.
        
        Args:
            ax: Matplotlib axis
            data: DataFrame with OHLCV columns
            x_offset: X-axis offset for positioning
            is_predicted: Use predicted styling if True
            show_volume: Whether to render volume bars
        """
        colors = self.predicted_colors if is_predicted else self.actual_colors
        alpha = 0.6 if is_predicted else 1.0
        linestyle = '--' if is_predicted else '-'
        linewidth = 1.5 if is_predicted else 1.0
        
        for i, (idx, row) in enumerate(data.iterrows()):
            x = x_offset + i
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Determine candle color
            is_bullish = close_price >= open_price
            color = colors['bullish'] if is_bullish else colors['bearish']
            
            # Draw wick (high-low line)
            ax.plot(
                [x, x], [low_price, high_price],
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=1
            )
            
            # Draw candle body
            body_bottom = min(open_price, close_price)
            body_height = abs(close_price - open_price)
            
            if body_height > 0:  # Regular candle
                rect = Rectangle(
                    (x - 0.4, body_bottom), 0.8, body_height,
                    facecolor=color, edgecolor=color,
                    alpha=alpha, linewidth=linewidth,
                    linestyle=linestyle, zorder=2
                )
                ax.add_patch(rect)
            else:  # Doji
                ax.plot(
                    [x - 0.4, x + 0.4], [open_price, open_price],
                    color=color, linewidth=linewidth * 1.5,
                    linestyle=linestyle, alpha=alpha, zorder=2
                )
        
        # Render volume if requested
        if show_volume and 'Volume' in data.columns:
            self._render_volume(ax, data, x_offset, is_predicted, alpha)
    
    def _render_volume(
        self,
        ax,
        data: pd.DataFrame,
        x_offset: int,
        is_predicted: bool,
        alpha: float
    ):
        """Render volume bars at bottom of chart."""
        # Create volume subplot at bottom 20%
        colors = self.predicted_colors if is_predicted else self.actual_colors
        
        max_volume = data['Volume'].max()
        if max_volume == 0:
            return
        
        for i, (idx, row) in enumerate(data.iterrows()):
            x = x_offset + i
            volume = row['Volume']
            is_bullish = row['Close'] >= row['Open']
            color = colors['bullish'] if is_bullish else colors['bearish']
            
            # Normalize volume to 0.15 of price range for visualization
            vol_height = (volume / max_volume) * 0.15
            
            # Note: This is simplified - in full impl would use twinx() for proper scaling
    
    def render_comparison(
        self,
        actual_df: pd.DataFrame,
        predicted_df: pd.DataFrame,
        output_path: Union[str, Path],
        title: str = "Actual vs Predicted",
        mode: str = "side_by_side"
    ):
        """
        Render comparison between actual and predicted candles.
        
        Args:
            actual_df: Actual OHLCV data
            predicted_df: Predicted OHLCV data
            output_path: Path to save image
            title: Chart title
            mode: 'side_by_side' or 'overlay'
        """
        if mode == "side_by_side":
            self._render_side_by_side(actual_df, predicted_df, output_path, title)
        elif mode == "overlay":
            self._render_overlay(actual_df, predicted_df, output_path, title)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _render_side_by_side(
        self,
        actual_df: pd.DataFrame,
        predicted_df: pd.DataFrame,
        output_path: Union[str, Path],
        title: str
    ):
        """Render side-by-side comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size, dpi=self.dpi)
        
        # Left: Actual
        self.render_candlesticks(ax1, actual_df, is_predicted=False, show_volume=False)
        ax1.set_title("Actual", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Candle Index")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Right: Predicted
        self.render_candlesticks(ax2, predicted_df, is_predicted=True, show_volume=False)
        ax2.set_title("Predicted", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Candle Index")
        ax2.set_ylabel("Price")
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Match Y-axis scales
        all_prices = pd.concat([
            actual_df[['High', 'Low']],
            predicted_df[['High', 'Low']]
        ])
        y_min = all_prices.min().min() * 0.995
        y_max = all_prices.max().max() * 1.005
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        # Main title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved side-by-side comparison to {output_path}")
    
    def _render_overlay(
        self,
        actual_df: pd.DataFrame,
        predicted_df: pd.DataFrame,
        output_path: Union[str, Path],
        title: str
    ):
        """Render overlay comparison."""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Render actual candles first
        self.render_candlesticks(ax, actual_df, x_offset=0, is_predicted=False, show_volume=False)
        
        # Render predicted candles after actual (continuing timeline)
        x_offset = len(actual_df)
        self.render_candlesticks(ax, predicted_df, x_offset=x_offset, is_predicted=True, show_volume=False)
        
        # Add vertical separator line
        ax.axvline(x=x_offset - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Prediction Start')
        
        # Configure axis
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Candle Index")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.actual_colors['bullish'], label='Actual Bullish'),
            Patch(facecolor=self.actual_colors['bearish'], label='Actual Bearish'),
            Patch(facecolor=self.predicted_colors['bullish'], alpha=0.6, label='Predicted Bullish'),
            Patch(facecolor=self.predicted_colors['bearish'], alpha=0.6, label='Predicted Bearish')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved overlay comparison to {output_path}")
    
    def render_prediction_only(
        self,
        predicted_df: pd.DataFrame,
        output_path: Union[str, Path],
        title: str = "Predicted Candles"
    ):
        """
        Render only predicted candles.
        
        Args:
            predicted_df: Predicted OHLCV data
            output_path: Path to save image
            title: Chart title
        """
        fig, ax = plt.subplots(figsize=(self.fig_size[0] / 2, self.fig_size[1]), dpi=self.dpi)
        
        self.render_candlesticks(ax, predicted_df, is_predicted=True, show_volume=False)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Candle Index")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved predicted chart to {output_path}")
