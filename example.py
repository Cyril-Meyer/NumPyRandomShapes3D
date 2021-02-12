import numpy as np
import nprs
import NumPyDraw.view as npdview

print(nprs.dtype)

X = nprs.random_image((127, 128, 129), shapes_number=16, shapes_max_size=(63, 64, 65))
npdview.show_stack(X, interval=25, axis=0, vmin=0, vmax=255)