# ray_tracer.py
import numpy as np
from objects import sphere
import math
HEIGHT = 400
WIDTH = 400

aspect_ratio = WIDTH / HEIGHT
BG_COLOR = [255,255,255]

vp_height = 2
vp_width = vp_height*aspect_ratio

focal_length = 1

cam = np.array([0,0,0])
light_dir = np.array([1,1,1]) / np.linalg.norm(np.array([1,1,1]))

my_sphere = sphere([0,0,-2], 0.5, [255,0,0])

pixel_values = np.zeros((HEIGHT, WIDTH, 3), dtype = int)

def ray_color(cam, D):
    oc = cam - my_sphere.center
    # (O+tD-C)^2 =  R^2 qudratic expansion
    a = np.dot(D,D)
    b = 2*np.dot(oc, D)
    c = np.dot(oc, oc) - my_sphere.radius**2

    disc = b**2 - 4 * a * c 
    if disc >= 0: 
        t = (-b - math.sqrt(disc)) / (2 * a)
        P = cam + t*D

        N = (P-my_sphere.center) / np.linalg.norm(P-my_sphere.center)

        Intensity = max(0, np.dot(N,light_dir))

        return np.array(my_sphere.color) * Intensity

    return BG_COLOR 

def main_loop():
    for x in range(WIDTH):
        for y in range(HEIGHT):
            u = vp_width/WIDTH * (x+0.5) - vp_width / 2
            v = (vp_height - vp_height/HEIGHT * (y+0.5)) - vp_height / 2
            target = np.array([u,v,-focal_length])
            D = (target - cam) / np.linalg.norm(target - cam)
            
            
            pixel_values[y][x] = ray_color(cam, D)
            

def save_image():
    f = open("render.ppm", "w")
    f.write("P3\n")
    f.write(f"{WIDTH} {HEIGHT}\n")
    f.write("255\n")
    for y in range(HEIGHT):
        for x in range(WIDTH):
            for _ in range(3):
                r = pixel_values[y][x][0]
                g = pixel_values[y][x][1]
                b = pixel_values[y][x][2]

            f.write(f"{r} {g} {b}\n")
    f.close()



main_loop()
save_image()