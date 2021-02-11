from random import randint
import numpy as np
import skimage as sk
import skimage.morphology
import NumPyDraw.npd3d as npd3d


def add_random_spheroid(array, shapes_max_size, fill):
    """
    return the given array filled with a random spheroid.
    """
    center = randint(0, array.shape[0]-1), randint(0, array.shape[1]-1), randint(0, array.shape[2]-1)
    radius = randint(1, shapes_max_size[0]), randint(1, shapes_max_size[1]), randint(1, shapes_max_size[2]),
    rotxy = np.deg2rad(randint(0, 90))
    array[npd3d.spheroid_coordinate(array.shape, center, radius, rotxy)] = fill
    return array


def random_shapes(array, shapes_number=1, shapes_max_size=(None, None, None), fill_range=(1, 1)):
    """
    return the given array filled with random shapes.
    """
    sms = np.asarray(array.shape)
    for i in range(len(shapes_max_size)):
        if shapes_max_size[i] is not None:
            sms[i] = min(shapes_max_size[i], array.shape[i])
    shapes_max_size = tuple(sms)

    for _ in range(shapes_number):
        fill = randint(fill_range[0], fill_range[1])
        array = add_random_spheroid(array, shapes_max_size, fill)

    return array

