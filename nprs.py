from random import randint
import numpy as np
import cv2
import scipy.ndimage
import NumPyDraw.npd3d as npd3d
from perlin_numpy import generate_perlin_noise_3d
from perlin_numpy import generate_fractal_noise_3d

dtype = np.uint8


def random_spheroid(array_shape, shapes_max_size):
    """
    return an array filled with a random spheroid.
    """
    array = np.zeros(array_shape, dtype=dtype)
    center = randint(0, array.shape[0]-1), randint(0, array.shape[1]-1), randint(0, array.shape[2]-1)
    radius = randint(1, shapes_max_size[0]), randint(1, shapes_max_size[1]), randint(1, shapes_max_size[2])
    rot = np.deg2rad(randint(0, 90)), np.deg2rad(randint(0, 90)), np.deg2rad(randint(0, 90))

    array[npd3d.spheroid_coordinate(array.shape, center, radius, rot)] = 1

    return array


def add_random_spheroid(array, shapes_max_size, fill):
    """
    return the given array filled with a random spheroid.
    """
    center = randint(0, array.shape[0]-1), randint(0, array.shape[1]-1), randint(0, array.shape[2]-1)
    radius = randint(1, shapes_max_size[0]), randint(1, shapes_max_size[1]), randint(1, shapes_max_size[2])
    rot = np.deg2rad(randint(0, 90)), np.deg2rad(randint(0, 90)), np.deg2rad(randint(0, 90))

    array[npd3d.spheroid_coordinate(array.shape, center, radius, rot)] = fill
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


def random_shape_multi_resolution(array_shape, shapes_max_size, resolution=1, iteration=10):
    """
    return an array filled with a random shape using multi resolution transformation.
    precondition : array_shape and shapes_max_size must be multiple of 2^resolution
    """
    array_shape_multi = tuple(np.asarray(array_shape) // 2**resolution)
    shapes_max_size_multi = tuple(np.asarray(shapes_max_size) // 2**resolution)

    array = np.zeros(array_shape_multi, dtype=dtype)
    add_random_spheroid(array, shapes_max_size_multi, 1)

    for i in range(resolution+4):
        for _ in range(iteration):
            kernel = np.random.randint(2, size=(5, 5), dtype=np.uint8)
            array = cv2.dilate(array, kernel, iterations=1)
            kernel = np.random.randint(2, size=(5, 5), dtype=np.uint8)
            array = cv2.erode(array, kernel, iterations=1)

        if i < resolution:
            array = np.repeat(np.repeat(np.repeat(array, 2, axis=0), 2, axis=1), 2, axis=2)

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

        array = np.moveaxis(array, [0, 1], [1, 2])

    while array.shape != array_shape:
        array = np.moveaxis(array, [0, 1], [1, 2])

    return array


def random_image(array_shape, shapes_number, shapes_max_size=(None, None, None)):
    """
    return an image filled with random shapes with noise and texture.
    to avoid using to much parameters in the function, most of them are fixed at the start of the function.
    we recommend to avoid shapes_max_size > (256, 256, 256) or to disable perlin / fractal noise.
    """
    fill_range = (0, 255)
    background_noise_mean = 128
    background_noise_std = 32

    texture_noise_res_min = 2
    texture_noise_res_max = 16
    texture_noise_amp = 42
    tnrmin = texture_noise_res_min
    tnrmax = texture_noise_res_max

    gaussian_noise_mean = 0
    gaussian_noise_std = 10

    # check shapes_max_size argument
    sms = np.asarray(array_shape)
    for i in range(len(shapes_max_size)):
        if shapes_max_size[i] is not None:
            sms[i] = min(shapes_max_size[i], array_shape[i])
    shapes_max_size = tuple(sms)

    # background noise
    array = (np.random.standard_normal(array_shape) * background_noise_std + background_noise_mean).astype(np.uint8)
    array = scipy.ndimage.gaussian_filter(array, 0.5)

    # shapes
    for _ in range(shapes_number):
        # create new shape
        fill = randint(fill_range[0], fill_range[1])
        mask = random_shape(array.shape, shapes_max_size, iteration=10)
        # mask = random_shape_multi_resolution(array.shape, shapes_max_size, resolution=2, iteration=10)
        if np.sum(mask) == 0:
            continue

        # add perlin / fractal noise to create texture for the shape
        shape = np.copy(mask)

        # get mask bounding box for texture and noise
        x, y, z = np.where(mask)

        x_min, y_min, z_min = x.min(), y.min(), z.min()
        x_max, y_max, z_max = x.max(), y.max(), z.max()
        x_size, y_size, z_size = x_max - x_min, y_max - y_min, z_max - z_min

        # apply noise to mask
        noise_res_x, noise_res_y, noise_res_z = randint(tnrmin, tnrmax), randint(tnrmin, tnrmax), randint(tnrmin, tnrmax)

        x_size, y_size, z_size = ((x_size//noise_res_x)+1)*noise_res_x,\
                                 ((y_size//noise_res_y)+1)*noise_res_y,\
                                 ((z_size//noise_res_z)+1)*noise_res_z

        texture_noise = generate_perlin_noise_3d(
            (x_size, y_size, z_size), (noise_res_x, noise_res_y, noise_res_z), tileable=(False, False, False)
        )

        shape_subpart = shape[x_min:x_min+x_size, y_min:y_min+y_size, z_min:z_min+z_size].shape

        shape[x_min:x_min+x_size, y_min:y_min+y_size, z_min:z_min+z_size] = \
            np.clip(((texture_noise[0:shape_subpart[0], 0:shape_subpart[1], 0:shape_subpart[2]] * 2) * texture_noise_amp) + fill, 0, 255)

        # add gaussian noise
        gaussian_noise = np.random.normal(gaussian_noise_mean, gaussian_noise_std, shape.shape)
        shape[mask == 1] = np.clip(shape[mask == 1] + gaussian_noise[mask == 1], 0, 255)

        array[mask == 1] = shape[mask == 1]

        array = np.moveaxis(array, [0, 1], [1, 2])

    while array.shape != array_shape:
        array = np.moveaxis(array, [0, 1], [1, 2])

    return array
