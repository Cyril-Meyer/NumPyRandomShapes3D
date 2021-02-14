import random
import time
import numpy as np
import nprs
import NumPyDraw.view as npdview

print(nprs.dtype)

t0 = time.time()
# X = nprs.random_spheroid((64, 64, 64), (32, 32, 32))*255
# X = nprs.random_shape_multi_resolution((256, 256, 256), shapes_max_size=(128, 128, 128))*255
# X = nprs.random_shapes((64, 64, 64), shapes_number=16, shapes_max_size=(32, 32, 32), fill_range=(0, 255))
X = nprs.random_image((128, 256, 256), shapes_number=64, shapes_max_size=(64, 128, 128))
t1 = time.time()
print("time:", t1 - t0)

npdview.show_stack(X, interval=25, axis=0, vmin=0, vmax=255)
npdview.gif_stack(X, "example/random_3D_image.gif", interval=25, axis=0, vmin=0, vmax=255)
