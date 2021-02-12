import numpy as np
import nprs
import NumPyDraw.view as npdview

print(nprs.dtype)

# X = nprs.random_spheroid((64, 64, 64), (32, 32, 32))*255
# X = nprs.random_shape_multi_resolution((256, 256, 256), shapes_max_size=(128, 128, 128))*255
# X = nprs.random_shapes((64, 64, 64), shapes_number=16, shapes_max_size=(32, 32, 32), fill_range=(0, 255))
X = nprs.random_image((64, 128, 160), shapes_number=16, shapes_max_size=(32, 64, 64))

npdview.show_stack(X, interval=25, axis=0, vmin=0, vmax=255)