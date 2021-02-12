import numpy as np
import nprs
import NumPyDraw.view as npdview

print(nprs.dtype)

X = nprs.random_shapes((128, 128, 128), shapes_number=16, shapes_max_size=(64, 64, 64), fill_range=(1, 4))
npdview.show_stack(X, interval=25, axis=2, vmin=0, vmax=4)