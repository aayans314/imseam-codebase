import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import os
import math

def get_weight_corners(led_pos, pd_center, w_pd, lz, mask_z, pd_z):
    px, py = pd_center[0], pd_center[1]
    pd_corners = [
        np.array([px - w_pd/2, py - w_pd/2, pd_z]),
        np.array([px + w_pd/2, py - w_pd/2, pd_z]),
        np.array([px + w_pd/2, py + w_pd/2, pd_z]),
        np.array([px - w_pd/2, py + w_pd/2, pd_z])
    ]
    mask_corners = []
    for corner in pd_corners:
        direction = corner - led_pos
        t = (mask_z - lz) / direction[2]
        mask_corner = led_pos + t * direction
        mask_corners.append(mask_corner)
    return np.array(mask_corners), pd_corners

def draw_plane(ax, z, width, height, color='gray', alpha=0.3, edge_color='white'):
    corners = [
        [-width/2, -height/2, z],
        [width/2, -height/2, z],
        [width/2, height/2, z],
        [-width/2, height/2, z]
    ]
    poly = Poly3DCollection([corners], facecolors=color, alpha=alpha, edgecolors=edge_color, linewidths=0.5)
    ax.add_collection3d(poly)

def main():
    plt.style.use('dark_background')
    
    # 1. Load weights
    weight_path = 'mnist_64x64_weights.pth'
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found.")
        return
    st = torch.load(weight_path, map_location='cpu')
    
    W1_raw = st['fc1.weight']
    W1_relu = F.relu(W1_raw).numpy()
    W1_max = np.percentile(W1_relu[W1_relu > 0], 98) if len(W1_relu[W1_relu > 0]) > 0 else 1.0
    W1 = np.clip(W1_relu / W1_max, 0, 1)
    
    W2_raw = st['readout.weight']
    W2_relu = F.relu(W2_raw).numpy()
    W2_max = np.percentile(W2_relu[W2_relu > 0], 98) if len(W2_relu[W2_relu > 0]) > 0 else 1.0
    W2 = np.clip(W2_relu / W2_max, 0, 1)

    # 2. Get one MNIST sample
    print("Loading MNIST sample...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((7, 7), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.Pad((0, 0, 1, 1), fill=0)
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    sample_img, label = dataset[0] 
    X = sample_img.view(64).numpy()

    # 3. Simulate Forward Pass mathematically
    out1 = W1 @ X
    pos_signals = out1[:32]
    neg_signals = out1[32:]
    diff = pos_signals - neg_signals
    hidden_acts = np.maximum(0, diff) # ReLU
    
    H_max = np.max(hidden_acts) if np.max(hidden_acts) > 0 else 1.0
    H_norm = hidden_acts / H_max
    
    out2 = W2 @ hidden_acts
    O_max = np.max(out2) if np.max(out2) > 0 else 1.0
    O_norm = out2 / O_max

    # 4. Physical Parameters
    s_led = 3.5
    s_pd = 3.0
    area_pd = 1.54
    w_pd = math.sqrt(area_pd)
    
    d1, d2 = 3.0, 34.5
    
    z_led1 = 0.0
    z_mask1 = z_led1 + d1
    z_pd1 = z_led1 + d1 + d2
    
    z_led2 = z_pd1
    z_mask2 = z_led2 + d1
    z_pd2 = z_led2 + d1 + d2
    
    # Coordinate grids
    l1_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l1_led_y = np.linspace(3.5*s_led, -3.5*s_led, 8)
    
    l1_pd_x = np.linspace(-3.5*s_pd, 3.5*s_pd, 8)
    l1_pd_y = np.linspace(3.5*s_pd, -3.5*s_pd, 8)
    
    l2_led_x = np.linspace(-3.5*s_led, 3.5*s_led, 8)
    l2_led_y = np.linspace(1.5*s_led, -1.5*s_led, 4)
    
    l2_pd_x = np.linspace(-2.0*s_pd, 2.0*s_pd, 5)
    l2_pd_y = np.linspace(0.5*s_pd, -0.5*s_pd, 2)

    # Calculate bounding box of the entire PD arrays
    l1_corners = [
        np.array([min(l1_pd_x)-w_pd/2, min(l1_pd_y)-w_pd/2, z_pd1]),
        np.array([max(l1_pd_x)+w_pd/2, min(l1_pd_y)-w_pd/2, z_pd1]),
        np.array([max(l1_pd_x)+w_pd/2, max(l1_pd_y)+w_pd/2, z_pd1]),
        np.array([min(l1_pd_x)-w_pd/2, max(l1_pd_y)+w_pd/2, z_pd1])
    ]
    
    l2_corners = [
        np.array([min(l2_pd_x)-w_pd/2, min(l2_pd_y)-w_pd/2, z_pd2]),
        np.array([max(l2_pd_x)+w_pd/2, min(l2_pd_y)-w_pd/2, z_pd2]),
        np.array([max(l2_pd_x)+w_pd/2, max(l2_pd_y)+w_pd/2, z_pd2]),
        np.array([min(l2_pd_x)-w_pd/2, max(l2_pd_y)+w_pd/2, z_pd2])
    ]

    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Optoelectronic 2-Layer Neural Network Hardware Layout", color='white', fontsize=16)

    ax.set_box_aspect((30, 30, 75))

    weight_patches = []
    weight_colors = []
    pd_patches = []
    pd_colors = []
    
    print("Tracing rays for Layer 1...")
    
    for i in range(8):
        for j in range(8):
            led_idx = i * 8 + j
            lx, ly = l1_led_x[j], l1_led_y[i]
            led_pos = np.array([lx, ly, z_led1])
            
            intensity = X[led_idx]
            
            # Draw LED as green, brightness controlled by alpha
            if intensity > 0:
                ax.scatter(lx, ly, z_led1, color=(0, 1, 0, intensity), s=30, zorder=5)
            else:
                ax.scatter(lx, ly, z_led1, color=(0, 1, 0, 0.1), s=30, zorder=5) # faint resting LED
            
            # Draw the 4 bounding rays for the entire target array from THIS LED
            if intensity > 0.1: # Only draw rays for active LEDs so it's not totally solid
                for corner in l1_corners:
                    ax.plot([lx, corner[0]], [ly, corner[1]], [z_led1, corner[2]], 
                            color='yellow', alpha=0.04 * intensity, linewidth=0.8)
            
            for pr in range(8):
                for pc in range(8):
                    pd_idx = pr * 8 + pc
                    px, py = l1_pd_x[pc], l1_pd_y[pr]
                    pd_center = np.array([px, py, z_pd1])
                    
                    val = W1[pd_idx, led_idx]
                    m_corners, p_corners = get_weight_corners(led_pos, pd_center, w_pd, z_led1, z_mask1, z_pd1)
                    
                    if val > 0.05:
                        weight_patches.append(m_corners)
                        alpha = val * 0.8 + 0.1
                        weight_colors.append([0.8, 0.8, 0.8, alpha])

    for pr in range(8):
        for pc in range(8):
            px, py = l1_pd_x[pc], l1_pd_y[pr]
            p_corners = [
                np.array([px - w_pd/2, py - w_pd/2, z_pd1]),
                np.array([px + w_pd/2, py - w_pd/2, z_pd1]),
                np.array([px + w_pd/2, py + w_pd/2, z_pd1]),
                np.array([px - w_pd/2, py + w_pd/2, z_pd1])
            ]
            pd_patches.append(p_corners)
            pd_colors.append([0.2, 0.2, 0.2, 0.8])
                        
    print("Tracing rays for Layer 2...")
    
    for i in range(4):
        for j in range(8):
            led_idx = i * 8 + j
            lx, ly = l2_led_x[j], l2_led_y[i]
            led_pos = np.array([lx, ly, z_led2])
            
            intensity = H_norm[led_idx]
            
            # Draw LED as green
            if intensity > 0:
                ax.scatter(lx, ly, z_led2, color=(0, 1, 0, intensity), s=30, zorder=5)
            else:
                ax.scatter(lx, ly, z_led2, color=(0, 1, 0, 0.1), s=30, zorder=5)
            
            if intensity > 0.1:
                for corner in l2_corners:
                    ax.plot([lx, corner[0]], [ly, corner[1]], [z_led2, corner[2]], 
                            color='cyan', alpha=0.08 * intensity, linewidth=0.8)
            
            for pr in range(2):
                for pc in range(5):
                    pd_idx = pr * 5 + pc
                    px, py = l2_pd_x[pc], l2_pd_y[pr]
                    pd_center = np.array([px, py, z_pd2])
                    
                    val = W2[pd_idx, led_idx]
                    m_corners, p_corners = get_weight_corners(led_pos, pd_center, w_pd, z_led2, z_mask2, z_pd2)
                    
                    if val > 0.05:
                        weight_patches.append(m_corners)
                        alpha = val * 0.8 + 0.1
                        weight_colors.append([0.8, 0.8, 0.8, alpha])
                        
    for pr in range(2):
        for pc in range(5):
            pd_idx = pr * 5 + pc
            px, py = l2_pd_x[pc], l2_pd_y[pr]
            intensity = O_norm[pd_idx]
            p_corners = [
                np.array([px - w_pd/2, py - w_pd/2, z_pd2]),
                np.array([px + w_pd/2, py - w_pd/2, z_pd2]),
                np.array([px + w_pd/2, py + w_pd/2, z_pd2]),
                np.array([px - w_pd/2, py + w_pd/2, z_pd2])
            ]
            pd_patches.append(p_corners)
            pd_colors.append((0, 1, 0, intensity)) # Green output PDs too
            
    print("Rendering 3D Plot...")
    
    draw_plane(ax, z_led1, 30, 30, color='#1e2a38', alpha=0.5, edge_color='#4CAF50')
    draw_plane(ax, z_mask1, 30, 30, color='#4FC3F7', alpha=0.1, edge_color='cyan')
    draw_plane(ax, z_pd1, 30, 30, color='#1e2a38', alpha=0.5, edge_color='#4CAF50')
    draw_plane(ax, z_mask2, 30, 30, color='#4FC3F7', alpha=0.1, edge_color='cyan')
    draw_plane(ax, z_pd2, 30, 30, color='#1e2a38', alpha=0.5, edge_color='#4CAF50')

    poly = Poly3DCollection(weight_patches, facecolors=weight_colors, edgecolors='none')
    ax.add_collection3d(poly)
    
    poly_pd = Poly3DCollection(pd_patches, facecolors=pd_colors, edgecolors='white', linewidths=0.5)
    ax.add_collection3d(poly_pd)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, z_pd2 + 5)
    ax.set_xlabel('X (mm)', color='white')
    ax.set_ylabel('Y (mm)', color='white')
    ax.set_zlabel('Z (mm)', color='white')
    ax.tick_params(colors='white')
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    import matplotlib.animation as animation
    print("Generating 3D rotation GIF animation...")
    def update(frame):
        ax.view_init(elev=15, azim=frame)
        return fig,
        
    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 10), interval=100)
    anim.save('C:/Users/HP/.gemini/antigravity-ide/brain/e6af67b9-21b7-4c7c-b7bf-c42a371e207b/3d_projection.gif', writer='pillow')
    print("Done!")

if __name__ == "__main__":
    main()
