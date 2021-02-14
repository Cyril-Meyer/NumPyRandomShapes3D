import random
import time
import numpy as np
import nprs
import NumPyDraw.view as npdview

print(nprs.dtype)

t0 = time.time()
RSpheroid = nprs.random_spheroid((64, 64, 64), (32, 32, 32))*255
t1 = time.time()
print("time:", t1 - t0)

t0 = time.time()
RShapes = nprs.random_shapes((64, 64, 64), shapes_number=32, shapes_max_size=(32, 32, 32), fill_range=(0, 255))
t1 = time.time()
print("time:", t1 - t0)

t0 = time.time()
RImage = nprs.random_image((128, 128, 128), shapes_number=64, shapes_max_size=(64, 64, 64))
t1 = time.time()
print("time:", t1 - t0)

npdview.show_stack(RSpheroid, interval=25, axis=0, vmin=0, vmax=255)
npdview.show_stack(RShapes, interval=25, axis=0, vmin=0, vmax=255)
npdview.show_stack(RImage, interval=25, axis=0, vmin=0, vmax=255)

npdview.gif_stack(RSpheroid, "example/random_3D_spheroid.gif", interval=25, axis=0, vmin=0, vmax=255)
npdview.gif_stack(RShapes, "example/random_3D_shapes.gif", interval=25, axis=0, vmin=0, vmax=255)
npdview.gif_stack(RImage, "example/random_3D_image.gif", interval=25, axis=0, vmin=0, vmax=255)
