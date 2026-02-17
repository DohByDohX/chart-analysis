
import torch
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.loss import VGGPerceptualLoss, VisionTraderLoss

def test_vgg_minimal():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("Initializing VisionTraderLoss...")
    start_time = time.time()
    criterion = VisionTraderLoss(perceptual_weight=1.0, ssim_weight=1.0).to(device)
    print(f"Initialized in {time.time() - start_time:.4f}s")
    
    batch_size = 16
    img_size = 512
    
    print(f"Creating dummy inputs ({batch_size}, 3, {img_size}, {img_size})...")
    pred = torch.randn(batch_size, 3, img_size, img_size).to(device)
    target = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print("Running VisionTraderLoss forward pass...")
    start_time = time.time()
    loss_dict = criterion(pred, target)
    torch.cuda.synchronize()
    print(f"Forward pass completed in {time.time() - start_time:.4f}s")
    print(f"Loss: {loss_dict['loss'].item()}")
    
    print("Running loop for 10 iters...")
    start_time = time.time()
    for i in range(10):
        loss_dict = criterion(pred, target)
    torch.cuda.synchronize()
    print(f"10 iters completed in {time.time() - start_time:.4f}s")

if __name__ == "__main__":
    try:
        test_vgg_minimal()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
