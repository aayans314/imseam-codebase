import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import os
import math
from PIL import Image

def extract_dithered_weights():
    d1_path = 'layer1_photomask_dithered.png'
    d2_path = 'layer2_photomask_dithered.png'
    
    if not os.path.exists(d1_path) or not os.path.exists(d2_path):
        raise FileNotFoundError("Dithered masks not found! Run dither_masks.py first.")
        
    print("Loading 1-Bit Binary Dithered Photomasks from disk...")
    # PIL loads '1' mode images as boolean arrays which cast to 0.0 and 1.0 floats. Dividing by 255 crushes them!
    M1_img = np.array(Image.open(d1_path)).astype(np.float32)
    M2_img = np.array(Image.open(d2_path)).astype(np.float32)

    s_led = 3.5
    s_pd = 3.0
    w_pd = math.sqrt(1.54)
    d1, d2 = 3.0, 34.5
    M = (d1 + d2) / d1
    
    dx_mask = 0.01
    
    l1_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l1_led_y = np.linspace(3.5*s_led, -3.5*s_led, 8)
    l1_pd_x = np.linspace(-3.5*s_pd, 3.5*s_pd, 8)
    l1_pd_y = np.linspace(3.5*s_pd, -3.5*s_pd, 8)
    
    l2_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l2_led_y = np.linspace(1.5*s_led, -1.5*s_led, 4)
    l2_pd_x = np.linspace(-2.0*s_pd, 2.0*s_pd, 5)
    l2_pd_y = np.linspace(0.5*s_pd, -0.5*s_pd, 2)

    def to_c(x): return int((x + 15.0) / dx_mask)
    def to_r(y): return int((15.0 - y) / dx_mask)

    print("Extracting physical optical weights from Layer 1 Dithered Mask...")
    W1_optical_dithered = np.zeros((64, 64), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            led_idx = i * 8 + j
            lx, ly = l1_led_x[j], l1_led_y[i]
            for pr in range(8):
                for pc in range(8):
                    pd_idx = pr * 8 + pc
                    px, py = l1_pd_x[pc], l1_pd_y[pr]
                    
                    mx1, mx2 = min(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M), max(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M)
                    my1, my2 = min(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M), max(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M)
                    
                    # Get the average transmission of the dithered pixels in this bounding box
                    box = M1_img[to_r(my2):to_r(my1), to_c(mx1):to_c(mx2)]
                    if box.size > 0:
                        W1_optical_dithered[pd_idx, led_idx] = np.mean(box)

    print("Extracting physical optical weights from Layer 2 Dithered Mask...")
    W2_optical_dithered = np.zeros((10, 32), dtype=np.float32)
    for i in range(4):
        for j in range(8):
            led_idx = i * 8 + j
            lx, ly = l2_led_x[j], l2_led_y[i]
            for pr in range(2):
                for pc in range(5):
                    pd_idx = pr * 5 + pc
                    px, py = l2_pd_x[pc], l2_pd_y[pr]
                    
                    mx1, mx2 = min(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M), max(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M)
                    my1, my2 = min(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M), max(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M)
                    
                    box = M2_img[to_r(my2):to_r(my1), to_c(mx1):to_c(mx2)]
                    if box.size > 0:
                        W2_optical_dithered[pd_idx, led_idx] = np.mean(box)
                        
    return W1_optical_dithered, W2_optical_dithered

def main():
    print("==============================================")
    print("   DITHERED OPTICAL ACCURACY EVALUATION")
    print("==============================================\n")
    
    # 1. Load math weights and biases
    weight_path = 'mnist_64x64_weights.pth'
    st = torch.load(weight_path, map_location='cpu')
    
    # We now use torch.abs() to reflect the fixed training parametrizations
    W1 = torch.abs(st['fc1.weight']).numpy()
    W1_max = np.max(W1) if np.max(W1) > 0 else 1.0
    W1_math = W1 / W1_max
    
    W2 = torch.abs(st['readout.weight']).numpy()
    W2_max = np.max(W2) if np.max(W2) > 0 else 1.0
    W2_math = W2 / W2_max
    
    # Load the new electronic biases
    bias1 = st['bias1'].numpy()
    bias2 = st['bias2'].numpy()

    # 2. Extract dithered physical weights
    W1_dith, W2_dith = extract_dithered_weights()

    # 3. Load MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((7, 7), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.Pad((0, 0, 1, 1), fill=0)
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # Let's evaluate on the first 1,000 images for a quick but solid accuracy test
    num_samples = 1000
    print(f"\nEvaluating on {num_samples} MNIST Test Images...")
    
    math_correct = 0
    dith_correct = 0
    
    for idx in range(num_samples):
        img, label = dataset[idx]
        X = img.view(64).numpy()
        
        # --- Math Forward Pass ---
        # Note: W1_math is scaled by W1_max. To make the bias mathematically correct relative to the scaled weights,
        # we must apply the bias directly or unscale. Since X is unitless here, and we just want to match PyTorch:
        # Actually, if we scaled W1 by 1/W1_max, we should scale the bias by the same amount so the ReLU threshold matches!
        out1_m = W1_math @ X
        hidden_m = np.maximum(0, out1_m[:32] - out1_m[32:] + (bias1 / W1_max))
        out2_m = W2_math @ hidden_m
        pred_m = np.argmax(out2_m + (bias2 / (W1_max * W2_max)))
        if pred_m == label:
            math_correct += 1
            
        # --- Dithered Optics Forward Pass ---
        out1_d = W1_dith @ X
        hidden_d = np.maximum(0, out1_d[:32] - out1_d[32:] + (bias1 / W1_max))
        out2_d = W2_dith @ hidden_d
        pred_d = np.argmax(out2_d + (bias2 / (W1_max * W2_max)))
        if pred_d == label:
            dith_correct += 1

    math_acc = (math_correct / num_samples) * 100.0
    dith_acc = (dith_correct / num_samples) * 100.0

    print("\n==============================================")
    print("                   RESULTS")
    print("==============================================")
    print(f"Ideal Mathematical Accuracy:      {math_acc:.2f}%")
    print(f"Physical Dithered Optics Accuracy: {dith_acc:.2f}%")
    print("==============================================")

if __name__ == "__main__":
    main()
