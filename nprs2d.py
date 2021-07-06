import random
from random import randint
import numpy as np
import cv2
import scipy.ndimage
from perlin_numpy import generate_perlin_noise_2d
from perlin_numpy import generate_fractal_noise_2d
import NumPyDraw.npd2d as npd2d

dtype = np.uint8


def random_ellipsoid(array_shape, shapes_min_size, shapes_max_size, rot_range=(-20, 20)):
    """
    return an array filled with a random ellipsoid.
    """
    array = np.zeros(array_shape, dtype=dtype)
    center = randint(0, array.shape[0]-1), randint(0, array.shape[1]-1)
    radius = randint(shapes_min_size[0], shapes_max_size[0]),\
             randint(shapes_min_size[1], shapes_max_size[1])
    rot = np.deg2rad(randint(rot_range[0], rot_range[1]))

    array[npd2d.ellipse_coordinate(array.shape, center, radius, rot)] = 1

    return array


def add_random_ellipsoid(array, shapes_min_size, shapes_max_size, fill, rot_range=(-20, 20)):
    """
    return the given array filled with a random ellipsoid.
    """
    center = randint(0, array.shape[0]-1), randint(0, array.shape[1]-1)
    radius = randint(shapes_min_size[0], shapes_max_size[0]),\
             randint(shapes_min_size[1], shapes_max_size[1])
    rot = np.deg2rad(randint(rot_range[0], rot_range[1]))

    array[npd2d.ellipse_coordinate(array.shape, center, radius, rot)] = fill
    return array


def random_shape(array_shape, shapes_min_size, shapes_max_size, iteration=10):
    """
    return an array filled with a random shape. 2D version.
    """
    array = np.zeros(array_shape, dtype=dtype)
    add_random_ellipsoid(array, shapes_min_size, shapes_max_size, 1)

    # dilation and erosion with random structuring element to create random shapes from spheroid
    # We change the axis order because the operation are 2D
    for _ in range(iteration):
        kernel = np.random.randint(2, size=(16, 16), dtype=np.uint8)
        array = cv2.dilate(array, kernel, iterations=1)
        kernel = np.random.randint(2, size=(16, 16), dtype=np.uint8)
        array = cv2.erode(array, kernel, iterations=1)

    return array


def elastic_deformation(image, alpha=None, sigma=None, random_state=None):
    """
    apply random elastic deformation
    function based on _elastic function from https://github.com/ShuangXieIrene/ssds.pytorch
    """
    if alpha is None:
        alpha = image.shape[0] * random.uniform(0.5, 2)
    if sigma is None:
        sigma = int(image.shape[0] * random.uniform(0.5, 1))
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx, dy = [cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1) * alpha, (sigma | 1, sigma | 1), 0) for _ in
              range(2)]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32), np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
    image = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_REFLECT)

    return image


def random_image(array_shape, shapes_number, shapes_min_size=(1, 1), shapes_max_size=(None, None)):
    """
    return an image filled with random shapes with noise and texture.
    to avoid using to much parameters in the function, most of them are fixed at the start of the function.
    """
    background_noise = True
    shape_texture = True
    shape_noise = True
    final_noise = True

    fill_range = (0, 255)
    background_noise_mean = 128
    background_noise_std = 32

    texture_noise_res_min = 2
    texture_noise_res_max = 16
    texture_noise_amp = 42
    tnrmin = texture_noise_res_min
    tnrmax = texture_noise_res_max

    shape_gaussian_noise_mean = 0
    shape_gaussian_noise_std = 10

    final_added_gaussian_noise_mean = 0
    final_added_gaussian_noise_std = 20

    # check and set shapes_max_size
    sms = np.asarray(array_shape)
    for i in range(len(shapes_max_size)):
        if shapes_max_size[i] is not None:
            sms[i] = min(shapes_max_size[i], array_shape[i])
    shapes_max_size = tuple(sms)

    # check and set shapes_min_size
    sms = np.asarray(shapes_max_size)
    for i in range(len(shapes_min_size)):
        if shapes_min_size[i] is not None:
            sms[i] = min(shapes_min_size[i], array_shape[i])
    shapes_min_size = tuple(sms)

    # background noise
    if background_noise:
        array = (np.random.standard_normal(array_shape) * background_noise_std + background_noise_mean).astype(dtype)
        array = scipy.ndimage.gaussian_filter(array, 0.5)
    else:
        array = np.zeros(array_shape, dtype=dtype)
    print(array.dtype)

    # add random shapes
    for _ in range(shapes_number):
        # create new shape
        fill = randint(fill_range[0], fill_range[1])
        mask = elastic_deformation(random_ellipsoid(array.shape, shapes_min_size, shapes_max_size))
        # mask = (elasticdeform.deform_random_grid(random_spheroid(array.shape, shapes_max_size), sigma=4, points=3) > 0) * 1
        # mask = random_shape_multi_resolution(array.shape, shapes_max_size, resolution=2, iteration=3)
        # mask = random_shape(array.shape, shapes_max_size, iteration=10)

        # skip if void shape
        if np.sum(mask) == 0:
            continue

        shape = np.copy(mask)

        if shape_texture:
            # add perlin / fractal noise to create texture for the shape
            # get mask bounding box for texture and noise
            x, y = np.where(mask)

            x_min, y_min = x.min(), y.min()
            x_max, y_max = x.max(), y.max()
            x_size, y_size = x_max - x_min, y_max - y_min

            # apply noise to mask
            noise_res_x, noise_res_y = randint(tnrmin, tnrmax), \
                                       randint(tnrmin, tnrmax)

            x_size, y_size = ((x_size//noise_res_x)+1)*noise_res_x,\
                            ((y_size//noise_res_y)+1)*noise_res_y

            try:
                texture_noise = generate_perlin_noise_2d(
                    (x_size, y_size), (noise_res_x, noise_res_y), tileable=(False, False)
                )

            except:
                print("generate_perlin_noise_2d failed, alternative used")
                sX, sY = randint(1, 50) / 100, randint(1, 50) / 100
                rX, rY = (x_size * sX) // 2, (y_size * sY) // 2
                arX, arY = np.arange(-rX - 1, rX + 1, sX), \
                           np.arange(-rY - 1, rY + 1, sY)
                xx, yy = np.meshgrid(arY, arX)
                texture_noise = (np.sin(xx)/2 + np.sin(yy)/2 + np.tanh(xx+yy)/2)/4
                texture_noise = texture_noise[0:x_size, 0:y_size]

            shape_subpart = shape[x_min:x_min+x_size, y_min:y_min+y_size].shape

            shape[x_min:x_min+x_size, y_min:y_min+y_size] = \
                np.clip(((texture_noise[0:shape_subpart[0], 0:shape_subpart[1]] * 2)
                         * texture_noise_amp) + fill, 0, 255)
        else:
            shape = shape * fill

        if shape_noise:
            # add gaussian noise
            gaussian_noise = np.random.normal(shape_gaussian_noise_mean, shape_gaussian_noise_std, shape.shape)
            shape[mask == 1] = np.clip(shape[mask == 1] + gaussian_noise[mask == 1], 0, 255)

        array[mask == 1] = shape[mask == 1]

    if final_noise:
        # add gaussian noise
        array = np.clip(array + np.random.normal(final_added_gaussian_noise_mean, final_added_gaussian_noise_std, array.shape), 0, 255)

    return array, mask
