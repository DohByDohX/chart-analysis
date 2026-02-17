"""
Loss functions for Vision-to-Vision training.
Implements Perceptual Loss (VGG19) and SSIM Loss with masked region weighting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class VisionTraderLoss(nn.Module):
    """
    Combined Perceptual + SSIM loss with specific focus on the masked region.
    
    Components:
    1. Perceptual Loss (VGG19): Captures high-level features/style.
    2. SSIM Loss: Captures structural consistency.
    3. Masked Weighting: Applies higher weight to the rightmost masked region
       where the actual prediction happens.
    """
    
    def __init__(
        self,
        perceptual_weight: float = 1.0,
        ssim_weight: float = 1.0,
        masked_region_weight: float = 10.0,
        masked_pixels: int = 20,  # 5 candles * 4px/candle
    ):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.masked_region_weight = masked_region_weight
        self.masked_pixels = masked_pixels
        
        # Initialize VGG Perceptual Loss
        self.vgg = VGGPerceptualLoss().eval()
        
        # Initialize SSIM Loss
        self.ssim = SSIMLoss(window_size=11)
        
        logger.info(f"VisionTraderLoss initialized:")
        logger.info(f"  Perceptual weight: {perceptual_weight}")
        logger.info(f"  SSIM weight: {ssim_weight}")
        logger.info(f"  Masked region weight: {masked_region_weight} (last {masked_pixels}px)")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        # 1. Global Loss (Input + Prediction)
        # Helps maintain context consistency
        if self.perceptual_weight > 0:
            global_perceptual = self.vgg(pred, target)
        else:
            global_perceptual = torch.tensor(0.0, device=pred.device)
            
        global_ssim = self.ssim(pred, target)
        
        # 2. Masked Region Loss (Prediction Only)
        # Forces focus on the missing candles
        # Last N pixels width
        pred_masked = pred[..., -self.masked_pixels:]
        target_masked = target[..., -self.masked_pixels:]
        
        if self.perceptual_weight > 0:
            masked_perceptual = self.vgg(pred_masked, target_masked)
        else:
            masked_perceptual = torch.tensor(0.0, device=pred.device)
            
        masked_ssim = self.ssim(pred_masked, target_masked)
        
        # 3. Combine
        # Total = Global + (MaskWeight * Masked)
        perceptual_total = global_perceptual + (self.masked_region_weight * masked_perceptual)
        ssim_total = global_ssim + (self.masked_region_weight * masked_ssim)
        
        total_loss = (self.perceptual_weight * perceptual_total) + \
                     (self.ssim_weight * ssim_total)
                     
        return {
            "loss": total_loss,
            "perceptual": global_perceptual.item(),
            "ssim": global_ssim.item(),
            "masked_perceptual": masked_perceptual.item(),
            "masked_ssim": masked_ssim.item()
        }


class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual loss using VGG19 features.
    Extracts features from relu1_2, relu2_2, relu3_4, relu4_4.
    """
    
    def __init__(self, resize=True):
        super().__init__()
        self.resize = resize
        
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Extract slices we want
        # relu1_2 (idx 3), relu2_2 (idx 8), relu3_4 (idx 17), relu4_4 (idx 26)
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:18]
        self.slice4 = vgg[18:27]
        
        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize input (assuming pred/target are [0, 1])
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            pred = F.interpolate(pred, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        x = pred
        y = target
        
        for slice_layer in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = slice_layer(x)
            y = slice_layer(y)
            loss += F.l1_loss(x, y)
            
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    Implementation based on Wang et al. (2004).
    """
    
    def __init__(self, window_size: int = 11, output_channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = output_channels
        self.window = self._create_window(window_size, output_channels)

    def _gaussian(self, window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size) - window_size // 2)**2 / (2 * sigma**2))
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return nn.Parameter(window, requires_grad=False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # SSIM returns similarity (1 is max), so Loss = 1 - SSIM
        return 1.0 - self._ssim(img1, img2)

    def _ssim(self, img1, img2):
        # Ensure window is on correct device
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        channel = self.channel
        window = self.window
        window_size = self.window_size
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()
