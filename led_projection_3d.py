import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches

def get_window_coordinates(led_pos, corners_final, d1, d2):
    """
    Given an LED position and the 4 corners of the final projection rectangle,
    calculate the 4 coordinates of the window on the z=d1 plane.
    """
    z_final = d1 + d2
    window_corners = []
    for corner in corners_final:
        direction = corner - led_pos
        t = d1 / z_final
        window_corner = led_pos + t * direction
        window_corners.append(window_corner)
    return np.array(window_corners)

def main():
    # Parameters from the paper (Supplementary Table 3)
    # to avoid crosstalk and overlapping windows:
    d1 = 3.0   # Distance from LED to weight mask (mm)
    d2 = 34.5  # Distance from weight mask to PD plane (mm)
    z_final = d1 + d2  # 37.5 mm
    
    # 8x8 LED Array
    led_spacing = 2.5  # mm
    led_x = np.linspace(-3.5*led_spacing, 3.5*led_spacing, 8)
    led_y = np.linspace(-3.5*led_spacing, 3.5*led_spacing, 8)
    LEDs = [np.array([x, y, 0]) for x in led_x for y in led_y]
    
    # Final common rectangle (PD array area)
    # 8x8 PDs with 2.5mm spacing = 20mm x 20mm
    W_x, W_y = 20.0, 20.0 
    corners_final = np.array([
        [-W_x/2, -W_y/2, z_final],
        [ W_x/2, -W_y/2, z_final],
        [ W_x/2,  W_y/2, z_final],
        [-W_x/2,  W_y/2, z_final]
    ])
    
    # Print out an example calculation
    example_led = LEDs[0]
    example_window = get_window_coordinates(example_led, corners_final, d1, d2)
    print("--- Example Window Coordinate Calculation ---")
    print(f"LED Position: {example_led}")
    print(f"Final Screen Rect Corners (z={z_final}):")
    for c in corners_final:
        print(f"  {c}")
    print(f"Calculated Window Coordinates (z={d1}):")
    for w in example_window:
        print(f"  {w}")
    print("---------------------------------------------")

    # Setup plots
    fig = plt.figure(figsize=(16, 8))
    
    # --- 3D Plot ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"3D Ray Tracing Visualization\n(d1={d1}mm, d2={d2}mm)")
    
    # Plot final screen
    final_rect = Poly3DCollection([corners_final], alpha=0.2, facecolors='cyan', edgecolors='blue')
    ax1.add_collection3d(final_rect)
    
    corner_leds = [
        np.array([led_x[0], led_y[0], 0]),
        np.array([led_x[0], led_y[-1], 0]),
        np.array([led_x[-1], led_y[0], 0]),
        np.array([led_x[-1], led_y[-1], 0])
    ]
    
    # --- 2D Plot for Window Plane ---
    ax2 = fig.add_subplot(122)
    ax2.set_title(f"2D View of Window Plane (Masks) at z={d1}mm\nNotice the windows no longer overlap!")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Process each LED
    for led in LEDs:
        # 1. Get window coordinates
        window_corners = get_window_coordinates(led, corners_final, d1, d2)
        
        # 2. Add to 3D plot
        ax1.scatter(led[0], led[1], led[2], color='red', s=15)
        window_poly = Poly3DCollection([window_corners], alpha=0.4, facecolors='gray', edgecolors='black', linewidths=0.5)
        ax1.add_collection3d(window_poly)
        
        # Plot rays only for the 4 corner LEDs so 3D plot is understandable
        is_corner = any(np.allclose(led, c_led) for c_led in corner_leds)
        if is_corner:
            for wc, fc in zip(window_corners, corners_final):
                ax1.plot([led[0], fc[0]], [led[1], fc[1]], [led[2], fc[2]], color='orange', alpha=0.5, linewidth=1)
                
        # 3. Add to 2D plot
        xs = window_corners[:, 0]
        ys = window_corners[:, 1]
        ax2.fill(xs, ys, alpha=0.5, facecolor='gray', edgecolor='black')
        ax2.scatter(led[0], led[1], color='red', s=10, zorder=5)

    # Format 3D plot
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-15, 15)
    ax1.set_zlim(0, z_final + 5)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.view_init(elev=15, azim=-60)
    
    # Format 2D plot
    ax2.set_aspect('equal')
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-15, 15)
    
    # Legend for 2D plot
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5),
                    patches.Patch(facecolor='gray', edgecolor='black', alpha=0.5)]
    ax2.legend(custom_lines, ['LED (x,y) location', 'Window (Mask) area'])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
