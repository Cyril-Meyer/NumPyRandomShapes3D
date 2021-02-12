from random import randint
import numpy as np
import cv2
import NumPyDraw.npd3d as npd3d

dtype = np.uint8


def random_spheroid(array_shape, shapes_max_size):
    """
    return an array filled with a random spheroid.
    """
    array = np.zeros(array_shape, dtype=dtype)
    center = randint(0, array.shape[0]-1), randint(0, array.shape[1]-1), randint(0, array.shape[2]-1)
    radius = randint(1, shapes_max_size[0]), randint(1, shapes_max_size[1]), randint(1, shapes_max_size[2]),
    rotxy = np.deg2rad(randint(0, 90))
    array[npd3d.spheroid_coordinate(array.shape, center, radius, rotxy)] = 1
    return array


def add_random_spheroid(array, shapes_max_size, fill):
    """
    return the given array filled with a random spheroid.
    """
    center = randint(0, array.shape[0]-1), randint(0, array.shape[1]-1), randint(0, array.shape[2]-1)
    radius = randint(1, shapes_max_size[0]), randint(1, shapes_max_size[1]), randint(1, shapes_max_size[2]),
    rotxy = np.deg2rad(randint(0, 90))
    array[npd3d.spheroid_coordinate(array.shape, center, radius, rotxy)] = fill
    return array


def random_shape(array_shape, shapes_max_size, iteration=10):
    """
    return an array filled with a random shape.
    """
    array = np.zeros(array_shape, dtype=dtype)
    add_random_spheroid(array, shapes_max_size, 1)

    # dilation and erosion with random structuring element to create random shapes from spheroid
    for _ in range(iteration):
        kernel = np.random.randint(2, size=(5, 5), dtype=np.uint8)
        array = cv2.dilate(array, kernel, iterations=1)
        kernel = np.random.randint(2, size=(5, 5), dtype=np.uint8)
        array = cv2.erode(array, kernel, iterations=1)

    return array


def random_shapes(array_shape, shapes_number=1, shapes_max_size=(None, None, None), fill_range=(1, 1)):
    """
    return an array filled with random shapes.
    """
    array = np.zeros(array_shape, dtype=dtype)

    sms = np.asarray(array_shape)
    for i in range(len(shapes_max_size)):
        if shapes_max_size[i] is not None:
            sms[i] = min(shapes_max_size[i], array_shape[i])
    shapes_max_size = tuple(sms)

    for _ in range(shapes_number):
        fill = randint(fill_range[0], fill_range[1])
        mask = random_shape(array.shape, shapes_max_size)
        array[mask == 1] = fill

    return array
