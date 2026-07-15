import numpy as np
import math #

class sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

HEIGHT = 400
WIDTH = 400

aspect_ratio = WIDTH / HEIGHT
BG_COLOR = np.array([255, 255, 255])

vp_height = 2.0
vp_width = vp_height * aspect_ratio
focal_length = 1.0

cam = np.array([0, 0, 0])
light_dir = np.array([1, 1, 1]) / np.linalg.norm(np.array([1, 1, 1]))

my_sphere = sphere([0, 0, -2], 0.5, [255, 0, 0])
pixel_values = np.zeros((HEIGHT, WIDTH, 3), dtype=int)

du_dx = vp_width / WIDTH
dv_dy = -vp_height / HEIGHT

def trace(D):
    """Single ray hit test. Returns color or None if miss."""
    oc = cam - my_sphere.center
    a = np.dot(D, D)
    b = 2 * np.dot(oc, D)
    c = np.dot(oc, oc) - my_sphere.radius**2
    disc = b**2 - 4 * a * c

    if disc < 0:
        return None

    t = (-b - math.sqrt(disc)) / (2 * a)
    P = cam + t * D
    N = (P - my_sphere.center) / np.linalg.norm(P - my_sphere.center)
    Intensity = max(0.0, np.dot(N, light_dir))
    return my_sphere.color * Intensity

def ray_color(D, D_dx, D_dy):
    """
    Uses the ray differential to cheaply generate 3 extra sub-pixel
    rays (no re-derivation of camera geometry needed) and averages
    them with the primary ray -> antialiased edges.
    """
    samples = [
        D,
        D + 0.5 * D_dx,
        D + 0.5 * D_dy,
        D + 0.5 * (D_dx + D_dy),
    ]

    hits = []
    for s in samples:
        s = s / np.linalg.norm(s)
        c = trace(s)
        hits.append(c if c is not None else BG_COLOR)

    return np.mean(hits, axis=0)

def main_loop():
    for y in range(HEIGHT):
        for x in range(WIDTH):
            u = vp_width / WIDTH * (x + 0.5) - vp_width / 2
            v = (vp_height - vp_height / HEIGHT * (y + 0.5)) - vp_height / 2

            target = np.array([u, v, -focal_length])
            D = (target - cam) / np.linalg.norm(target - cam)

            target_dx = np.array([u + du_dx, v, -focal_length])
            target_dy = np.array([u, v + dv_dy, -focal_length])

            D_dx = ((target_dx - cam) / np.linalg.norm(target_dx - cam)) - D
            D_dy = ((target_dy - cam) / np.linalg.norm(target_dy - cam)) - D

            pixel_values[y][x] = ray_color(D, D_dx, D_dy)

def save_image():
    with open("render_differential.ppm", "w") as f:
        f.write("P3\n")
        f.write(f"{WIDTH} {HEIGHT}\n")
        f.write("255\n")
        for y in range(HEIGHT):
            for x in range(WIDTH):
                r, g, b = pixel_values[y][x]
                f.write(f"{r} {g} {b}\n")

main_loop()
save_image()