import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import torch.nn.functional as F
import os

def main():
    # 1. Load weights
    weight_path = 'mnist_64x64_weights.pth'
    if not os.path.exists(weight_path):
        print(f"Error: Could not find {weight_path}")
        return
        
    state_dict = torch.load(weight_path, map_location='cpu')
    W1 = torch.abs(state_dict['fc1.weight']).numpy()
    W1_max = np.max(W1) if np.max(W1) > 0 else 1.0
    W1_optical = W1 / W1_max
    
    W2 = torch.abs(state_dict['readout.weight']).numpy()
    W2_max = np.max(W2) if np.max(W2) > 0 else 1.0
    W2_optical = W2 / W2_max
    
    # We use W1 for the primary photomask
    W = W1_optical

    # 2. Construct 2D photomask image
    # Each individual weight is represented by a block of pixels to be visible
    wt_size = 5 # 5x5 pixels per weight
    subwindow_size = 8 * wt_size # 40 pixels per sub-window
    
    # Calculate physical gaps
    # LED spacing = 2.5mm, d1 = 3.0mm, d2 = 34.5mm, M = 12.5
    # Target plane size = 20mm x 20mm
    # Window width = 20 / 12.5 = 1.6 mm
    # Distance between window centers = 2.5 * (1 - 1/12.5) = 2.3 mm
    # Gap between windows = 2.3 - 1.6 = 0.7 mm
    # If 1.6 mm = 40 pixels, then 0.1 mm = 2.5 pixels
    # Gap = 0.7 mm = 17.5 pixels -> let's round to 18 pixels
    gap_size = 18 
    
    total_size = 8 * subwindow_size + 7 * gap_size # 320 + 126 = 446 pixels
    
    # Background is black (opaque)
    mask_image = np.zeros((total_size, total_size))
    
    # 3D plot data setup
    d1, d2 = 3.0, 34.5
    z_final = d1 + d2
    led_spacing = 2.5
    
    led_x = np.linspace(-3.5*led_spacing, 3.5*led_spacing, 8)
    led_y = np.linspace(3.5*led_spacing, -3.5*led_spacing, 8) # top-to-bottom
    
    weight_patches_3d = []
    weight_colors_3d = []
    
    # Populate mask image and 3D patches
    for i in range(8): # row of LEDs
        for j in range(8): # col of LEDs
            led_idx = i * 8 + j # LED flattening index
            
            # The sub-window for this LED on the 2D image
            start_row = i * (subwindow_size + gap_size)
            start_col = j * (subwindow_size + gap_size)
            
            # Extract the 64 weights for this specific LED to the 64 PDs
            # W[out_idx, in_idx]. out_idx goes from 0 to 63 (the 8x8 PD grid)
            w_led = W[:, led_idx].reshape(8, 8) 
            
            # Tile it to make it visible
            w_tiled = np.kron(w_led, np.ones((wt_size, wt_size)))
            mask_image[start_row:start_row+subwindow_size, start_col:start_col+subwindow_size] = w_tiled
            
            # --- 3D Geometry Mapping ---
            lx = led_x[j]
            ly = led_y[i] 
            
            # Center of the sub-window on z=d1 plane
            center_x = lx * (1 - d1/z_final)
            center_y = ly * (1 - d1/z_final)
            
            # Physical width of the sub-window
            w_width = 20.0 * d1 / z_final # 1.6 mm
            w_step = w_width / 8.0
            
            # Top-left corner of the subwindow
            sx = center_x - w_width/2
            sy = center_y + w_width/2 
            
            for pr in range(8): # row of PDs
                for pc in range(8): # col of PDs
                    val = w_led[pr, pc]
                    
                    # corners of this specific weight pixel
                    px1 = sx + pc * w_step
                    px2 = px1 + w_step
                    py1 = sy - pr * w_step
                    py2 = py1 - w_step
                    
                    corners = [
                        [px1, py1, d1],
                        [px2, py1, d1],
                        [px2, py2, d1],
                        [px1, py2, d1]
                    ]
                    weight_patches_3d.append(corners)
                    
                    # Physical visualization: val=1 (100% transmission) -> transparent
                    # val=0 (0% transmission) -> opaque (black)
                    alpha = (1.0 - val) * 0.9 # Opaque parts have high alpha (dark), transparent parts have low alpha
                    weight_colors_3d.append([0.1, 0.1, 0.1, alpha])

    # Save 2D image
    plt.imsave('photomask.png', mask_image, cmap='gray')
    print("Saved 2D photomask to photomask.png")

    # Generate plots
    fig = plt.figure(figsize=(18, 8))
    
    # 2D Plot
    ax1 = fig.add_subplot(121)
    ax1.imshow(mask_image, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Generated 2D Optical Photomask\n(Values normalized 0-1, White = Transparent)")
    ax1.axis('off')
    
    # 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("3D Demonstration of Photomask & Ray Tracing")
    
    # Draw weight patches (the actual photomask in 3D)
    poly = Poly3DCollection(weight_patches_3d, facecolors=weight_colors_3d, edgecolors='none')
    ax2.add_collection3d(poly)
    
    # Draw LEDs
    for i in range(8):
        for j in range(8):
            ax2.scatter(led_x[j], led_y[i], 0, color='red', s=8)
            
    # Draw Final Screen
    W_x, W_y = 20.0, 20.0
    corners_final = np.array([
        [-W_x/2, -W_y/2, z_final],
        [ W_x/2, -W_y/2, z_final],
        [ W_x/2,  W_y/2, z_final],
        [-W_x/2,  W_y/2, z_final]
    ])
    final_rect = Poly3DCollection([corners_final], alpha=0.1, facecolors='cyan', edgecolors='blue')
    ax2.add_collection3d(final_rect)
    
    # Draw sample rays for the top-left LED to show how light projects through its specific sub-window
    lx, ly = led_x[0], led_y[0]
    led_idx = 0
    w_led = W[:, led_idx].reshape(8, 8)
    for pr in range(8):
        for pc in range(8):
            val = w_led[pr, pc]
            if val > 0.1: # Only trace rays for weights that actually let light through
                # PD center
                pd_x = -W_x/2 + (pc + 0.5) * (W_x/8)
                pd_y = W_y/2 - (pr + 0.5) * (W_y/8)
                ax2.plot([lx, pd_x], [ly, pd_y], [0, z_final], color='orange', alpha=val*0.2, linewidth=0.5)

    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-15, 15)
    ax2.set_zlim(0, z_final + 5)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.savefig('C:/Users/HP/.gemini/antigravity-ide/brain/e6af67b9-21b7-4c7c-b7bf-c42a371e207b/photomask_2d.png', bbox_inches='tight')
    print("Saved 2D photomask representation to photomask_2d.png")

if __name__ == "__main__":
    main()
