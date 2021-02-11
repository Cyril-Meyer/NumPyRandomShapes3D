import numpy as np
import nprs
import NumPyDraw.view as npdview

X = np.zeros((256, 256, 256), dtype=np.uint8)
nprs.random_shapes(X, shapes_number=16, shapes_max_size=(128, 128, 128), fill_range=(1, 4))
npdview.show_stack(X, interval=25, axis=2)
