import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import math
from PIL import Image

def main():
    print("Initializing Floyd-Steinberg Photomask Dithering Engine...")
    
    # 1. Load weights
    weight_path = 'mnist_64x64_weights.pth'
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found.")
        return
    st = torch.load(weight_path, map_location='cpu')
    W1 = torch.abs(st['fc1.weight']).numpy()
    W1_max = np.max(W1) if np.max(W1) > 0 else 1.0
    W1_optical = W1 / W1_max
    
    W2 = torch.abs(st['readout.weight']).numpy()
    W2_max = np.max(W2) if np.max(W2) > 0 else 1.0
    W2_optical = W2 / W2_max

    # 3. Physical Parameters
    s_led = 3.5
    s_pd = 3.0
    w_pd = math.sqrt(1.54) # 1.24 mm
    d1, d2 = 3.0, 34.5
    M = (d1 + d2) / d1 # 12.5
    
    # Use 10-micron resolution (3000 x 3000 pixels) for a high-quality 9-Megapixel printable mask
    dx = 0.01 
    size = int(30.0 / dx)
    
    l1_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l1_led_y = np.linspace(3.5*s_led, -3.5*s_led, 8)
    l1_pd_x = np.linspace(-3.5*s_pd, 3.5*s_pd, 8)
    l1_pd_y = np.linspace(3.5*s_pd, -3.5*s_pd, 8)
    
    l2_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l2_led_y = np.linspace(1.5*s_led, -1.5*s_led, 4)
    l2_pd_x = np.linspace(-2.0*s_pd, 2.0*s_pd, 5)
    l2_pd_y = np.linspace(0.5*s_pd, -0.5*s_pd, 2)

    def to_c(x): return int((x + 15.0) / dx)
    def to_r(y): return int((15.0 - y) / dx)

    # --- A. Fabricate Continuous Photomasks ---
    print("Generating Layer 1 Grayscale Photomask...")
    M1_img = np.zeros((size, size), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            led_idx = i * 8 + j
            lx, ly = l1_led_x[j], l1_led_y[i]
            for pr in range(8):
                for pc in range(8):
                    pd_idx = pr * 8 + pc
                    px, py = l1_pd_x[pc], l1_pd_y[pr]
                    val = W1_optical[pd_idx, led_idx]
                    if val > 0:
                        mx1, mx2 = min(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M), max(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M)
                        my1, my2 = min(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M), max(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M)
                        M1_img[to_r(my2):to_r(my1), to_c(mx1):to_c(mx2)] = val
                        
    print("Generating Layer 2 Grayscale Photomask...")
    M2_img = np.zeros((size, size), dtype=np.float32)
    for i in range(4):
        for j in range(8):
            led_idx = i * 8 + j
            lx, ly = l2_led_x[j], l2_led_y[i]
            for pr in range(2):
                for pc in range(5):
                    pd_idx = pr * 5 + pc
                    px, py = l2_pd_x[pc], l2_pd_y[pr]
                    val = W2_optical[pd_idx, led_idx]
                    if val > 0:
                        mx1, mx2 = min(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M), max(lx + (px - w_pd/2 - lx)/M, lx + (px + w_pd/2 - lx)/M)
                        my1, my2 = min(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M), max(ly + (py - w_pd/2 - ly)/M, ly + (py + w_pd/2 - ly)/M)
                        M2_img[to_r(my2):to_r(my1), to_c(mx1):to_c(mx2)] = val

    # --- B. Floyd-Steinberg Dithering ---
    print("Applying Floyd-Steinberg dithering algorithm to Layer 1...")
    # Convert from float [0.0, 1.0] to uint8 [0, 255]
    M1_uint8 = (M1_img * 255).astype(np.uint8)
    pil_img1 = Image.fromarray(M1_uint8)
    # The '1' mode in PIL automatically uses Floyd-Steinberg dithering to binarize!
    dithered_img1 = pil_img1.convert('1', dither=Image.FLOYDSTEINBERG)
    
    print("Applying Floyd-Steinberg dithering algorithm to Layer 2...")
    M2_uint8 = (M2_img * 255).astype(np.uint8)
    pil_img2 = Image.fromarray(M2_uint8)
    dithered_img2 = pil_img2.convert('1', dither=Image.FLOYDSTEINBERG)

    # --- C. Export Print-Ready Files ---
    print("Exporting finalized binary masks to disk...")
    dithered_img1.save('layer1_photomask_dithered.png')
    dithered_img2.save('layer2_photomask_dithered.png')

    # --- D. Rendering a Visual Sample ---
    # We will plot a small zoomed-in crop of the mask so the user can actually see the binary dithering pixels
    print("Generating preview of the binarization output...")
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Floyd-Steinberg Dithered Photomask (10μm Resolution)", fontsize=16)

    # Crop out a 150x150 pixel region (~1.5mm x 1.5mm) from the center to show the micro-pixels
    center_r, center_c = size//2, size//2
    crop_size = 150
    
    crop_continuous = M1_img[center_r-crop_size:center_r+crop_size, center_c-crop_size:center_c+crop_size]
    # Convert binary image back to numpy for plotting
    dithered_np1 = np.array(dithered_img1).astype(float)
    crop_dithered = dithered_np1[center_r-crop_size:center_r+crop_size, center_c-crop_size:center_c+crop_size]

    axes[0].imshow(crop_continuous, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title("Original Continuous Grayscale Weights", fontsize=16, color='cyan')
    axes[0].axis('off')

    axes[1].imshow(crop_dithered, cmap='gray', interpolation='nearest')
    axes[1].set_title("Floyd-Steinberg Binary Dithered Pixels", fontsize=16, color='lime')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    print("Dithering complete! Files saved.")

if __name__ == "__main__":
    main()
