import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import os
import math
from PIL import Image

def main():
    print("Initializing Binary Dithered Optical Simulation...")
    
    # 1. Load weights and biases
    weight_path = 'mnist_64x64_weights.pth'
    if not os.path.exists(weight_path):
        return
    st = torch.load(weight_path, map_location='cpu')
    
    W1 = torch.abs(st['fc1.weight']).numpy()
    W1_max = np.max(W1) if np.max(W1) > 0 else 1.0
    W1_math = W1 / W1_max
    
    W2 = torch.abs(st['readout.weight']).numpy()
    W2_max = np.max(W2) if np.max(W2) > 0 else 1.0
    W2_math = W2 / W2_max
    
    bias1 = st['bias1'].numpy()
    bias2 = st['bias2'].numpy()

    # 2. Get MNIST sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((7, 7), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.Pad((0, 0, 1, 1), fill=0)
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    sample_img, label = dataset[0] 
    X = sample_img.view(64).numpy()

    # ==========================================
    # PURE MATHEMATICAL FORWARD PASS
    # ==========================================
    math_out1 = W1_math @ X
    pos_signals_m = math_out1[:32]
    neg_signals_m = math_out1[32:]
    math_diff = pos_signals_m - neg_signals_m + (bias1 / W1_max)
    math_hidden = np.maximum(0, math_diff)
    
    math_out2 = W2_math @ math_hidden
    math_final = math_out2 + (bias2 / (W1_max * W2_max))
    pred_math = np.argmax(math_final)

    # ==========================================
    # LOAD DITHERED MASKS
    # ==========================================
    d1_path = 'layer1_photomask_dithered.png'
    d2_path = 'layer2_photomask_dithered.png'
    if not os.path.exists(d1_path) or not os.path.exists(d2_path):
        print("Dithered masks not found! Run dither_masks.py first.")
        return
        
    print("Loading 1-Bit Binary Dithered Photomasks from disk...")
    M1_img = np.array(Image.open(d1_path)).astype(np.float32)
    M2_img = np.array(Image.open(d2_path)).astype(np.float32)

    s_led = 3.5
    s_pd = 3.0
    w_pd = math.sqrt(1.54)
    d1, d2 = 3.0, 34.5
    M = (d1 + d2) / d1
    
    dx_mask = 0.01
    size_mask = int(30.0 / dx_mask)
    
    l1_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l1_led_y = np.linspace(3.5*s_led, -3.5*s_led, 8)
    l1_pd_x = np.linspace(-3.5*s_pd, 3.5*s_pd, 8)
    l1_pd_y = np.linspace(3.5*s_pd, -3.5*s_pd, 8)
    
    l2_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l2_led_y = np.linspace(1.5*s_led, -1.5*s_led, 4)
    l2_pd_x = np.linspace(-2.0*s_pd, 2.0*s_pd, 5)
    l2_pd_y = np.linspace(0.5*s_pd, -0.5*s_pd, 2)

    # --- MONTE CARLO PHYSICS THROUGH BINARY MASKS ---
    photons_per_led = 2000000 
    print(f"Firing {photons_per_led:,} photons per LED through the BINARY DITHERED masks...")
    
    opt_out1 = np.zeros(64)
    
    for i in range(8):
        for j in range(8):
            led_idx = i * 8 + j
            intensity = X[led_idx]
            if intensity <= 0: continue
            lx, ly = l1_led_x[j], l1_led_y[i]
            
            X_p = np.random.uniform(-12.0, 12.0, photons_per_led)
            Y_p = np.random.uniform(-12.0, 12.0, photons_per_led)
            
            X_m, Y_m = lx + (X_p - lx)/M, ly + (Y_p - ly)/M
            C_m, R_m = ((X_m + 15.0)/dx_mask).astype(np.int32), ((15.0 - Y_m)/dx_mask).astype(np.int32)
            
            valid_mask = (C_m >= 0) & (C_m < size_mask) & (R_m >= 0) & (R_m < size_mask)
            
            transmission = np.zeros(photons_per_led)
            transmission[valid_mask] = M1_img[R_m[valid_mask], C_m[valid_mask]]
            
            photon_energy = intensity * transmission
            
            for pr in range(8):
                for pc in range(8):
                    pd_idx = pr * 8 + pc
                    px, py = l1_pd_x[pc], l1_pd_y[pr]
                    
                    hit = (X_p >= px - w_pd/2) & (X_p <= px + w_pd/2) & (Y_p >= py - w_pd/2) & (Y_p <= py + w_pd/2)
                    opt_out1[pd_idx] += np.sum(photon_energy[hit])

    # Multiply by a normalization constant
    scale_factor1 = np.max(math_out1) / (np.max(opt_out1) if np.max(opt_out1) > 0 else 1)
    opt_out1_scaled = opt_out1 * scale_factor1
    
    opt_diff_scaled = opt_out1_scaled[:32] - opt_out1_scaled[32:] + (bias1 / W1_max)
    opt_hidden = np.maximum(0, opt_diff_scaled)

    opt_out2 = np.zeros(10)
    for i in range(4):
        for j in range(8):
            led_idx = i * 8 + j
            intensity = opt_hidden[led_idx]
            if intensity <= 0: continue
            lx, ly = l2_led_x[j], l2_led_y[i]
            
            X_p = np.random.uniform(-8.0, 8.0, photons_per_led)
            Y_p = np.random.uniform(-4.0, 4.0, photons_per_led)
            
            X_m, Y_m = lx + (X_p - lx)/M, ly + (Y_p - ly)/M
            C_m, R_m = ((X_m + 15.0)/dx_mask).astype(np.int32), ((15.0 - Y_m)/dx_mask).astype(np.int32)
            
            valid_mask = (C_m >= 0) & (C_m < size_mask) & (R_m >= 0) & (R_m < size_mask)
            transmission = np.zeros(photons_per_led)
            transmission[valid_mask] = M2_img[R_m[valid_mask], C_m[valid_mask]]
            
            photon_energy = intensity * transmission
            
            for pr in range(2):
                for pc in range(5):
                    pd_idx = pr * 5 + pc
                    px, py = l2_pd_x[pc], l2_pd_y[pr]
                    
                    hit = (X_p >= px - w_pd/2) & (X_p <= px + w_pd/2) & (Y_p >= py - w_pd/2) & (Y_p <= py + w_pd/2)
                    opt_out2[pd_idx] += np.sum(photon_energy[hit])

    scale_factor2 = np.max(math_out2) / (np.max(opt_out2) if np.max(opt_out2) > 0 else 1)
    opt_out2_scaled = opt_out2 * scale_factor2
    
    opt_final = opt_out2_scaled + (bias2 / (W1_max * W2_max))
    pred_opt = np.argmax(opt_final)

    # Normalize plotting
    math_out1 /= np.max(math_out1) if np.max(math_out1) > 0 else 1
    math_hidden /= np.max(math_hidden) if np.max(math_hidden) > 0 else 1
    math_final /= np.max(math_final) if np.max(math_final) > 0 else 1
    
    opt_out1_scaled /= np.max(opt_out1_scaled) if np.max(opt_out1_scaled) > 0 else 1
    opt_hidden /= np.max(opt_hidden) if np.max(opt_hidden) > 0 else 1
    opt_final /= np.max(opt_final) if np.max(opt_final) > 0 else 1

    def annotate_heatmap(ax, data, shape):
        data_2d = data.reshape(shape)
        threshold = np.max(data_2d) / 2
        for i in range(shape[0]):
            for j in range(shape[1]):
                val = data_2d[i, j]
                color = 'black' if val > threshold else 'white'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8, fontweight='bold')

    print("Rendering plots...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    fig.suptitle(f"Dithered Optical MVM Simulation | Math Pred: {pred_math} | Optics Pred: {pred_opt} | True: {label}", fontsize=20, color='white')

    # MATHEMATICS
    axes[0,0].imshow(X.reshape(8, 8), cmap='hot', interpolation='nearest')
    axes[0,0].set_title("Input Vector", fontsize=14, color='cyan')
    axes[0,0].set_ylabel("Mathematical\nModel", fontsize=16, fontweight='bold', color='cyan')
    annotate_heatmap(axes[0,0], X, (8, 8))
    
    axes[0,1].imshow(math_out1.reshape(8, 8), cmap='hot', interpolation='nearest')
    axes[0,1].set_title("Layer 1 Projection", fontsize=14, color='cyan')
    annotate_heatmap(axes[0,1], math_out1, (8, 8))
    
    axes[0,2].imshow(math_hidden.reshape(4, 8), cmap='hot', interpolation='nearest')
    axes[0,2].set_title("Hidden Layer Activations", fontsize=14, color='cyan')
    annotate_heatmap(axes[0,2], math_hidden, (4, 8))
    
    axes[0,3].imshow(math_final.reshape(2, 5), cmap='hot', interpolation='nearest')
    axes[0,3].set_title(f"Output Logits", fontsize=14, color='cyan')
    annotate_heatmap(axes[0,3], math_final, (2, 5))

    # BINARY DITHERED OPTICS
    axes[1,0].imshow(X.reshape(8, 8), cmap='hot', interpolation='nearest')
    axes[1,0].set_title("Input LED Intensities", fontsize=14, color='magenta')
    axes[1,0].set_ylabel("Dithered Optical\nSimulation", fontsize=16, fontweight='bold', color='magenta')
    annotate_heatmap(axes[1,0], X, (8, 8))
    
    axes[1,1].imshow(opt_out1_scaled.reshape(8, 8), cmap='hot', interpolation='nearest')
    axes[1,1].set_title("Layer 1 PD Irradiance", fontsize=14, color='magenta')
    annotate_heatmap(axes[1,1], opt_out1_scaled, (8, 8))
    
    axes[1,2].imshow(opt_hidden.reshape(4, 8), cmap='hot', interpolation='nearest')
    axes[1,2].set_title("Hidden Layer LED Intensities", fontsize=14, color='magenta')
    annotate_heatmap(axes[1,2], opt_hidden, (4, 8))
    
    axes[1,3].imshow(opt_final.reshape(2, 5), cmap='hot', interpolation='nearest')
    axes[1,3].set_title(f"Output PD Signals", fontsize=14, color='magenta')
    annotate_heatmap(axes[1,3], opt_final, (2, 5))

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('C:/Users/HP/.gemini/antigravity-ide/brain/e6af67b9-21b7-4c7c-b7bf-c42a371e207b/compare_math_optics_dithered.png', bbox_inches='tight')
    print("Saved Dithered Optics comparison to compare_math_optics_dithered.png")

if __name__ == "__main__":
    main()
