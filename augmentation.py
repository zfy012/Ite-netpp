import elasticdeform
import numpy as np
import SimpleITK as sitk
from numpy import ndarray
from scipy.ndimage import affine_transform, shift, rotate


def translate3d(img, seg):
    r"""
    Translate the 3d image respect one of the 3 axis chosen
    """
    alpha, beta, gamma = 0.5 * np.random.random_sample(3)
    x_shift = alpha * img.shape[0]
    y_shift = beta * img.shape[1]
    z_shift = gamma * img.shape[2]

    img_new = shift(img, [x_shift, y_shift, z_shift], order=3, mode='constant')
    seg_new = shift(seg, [x_shift, y_shift, z_shift], order=0, mode='constant')

    return img_new, seg_new


def flip3d(img, seg):
    r"""
    Flip the 3d image respect one of the 3 axis chosen randomly
    """
    img_flip, seg_flip = img, seg
    choice = np.random.randint(1, 3)
    if choice == 0:  # flip x
        img_flip, seg_flip = img[::-1, ...], seg[::-1, ...]
    elif choice == 1:  # flip y
        img_flip, seg_flip = img[:, ::-1, ...], seg[:, ::-1, ...]
    elif choice == 2:  # flip z
        img_flip, seg_flip = img[..., ::-1], seg[..., ::-1]

    return img_flip, seg_flip


def rotate3d(img, seg):
    alpha, beta, gamma = np.random.random_sample(3) * np.pi / 2
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    R_rot: ndarray = np.dot(np.dot(Rx, Ry), Rz)

    img_new = affine_transform(img, R_rot, offset=0, order=3, mode='constant', cval=0)
    seg_new = affine_transform(seg, R_rot, offset=0, order=0, mode='constant', cval=0)

    return img_new, seg_new


def brightness(img, seg):
    r"""
    Changing the brightness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for image.

    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]

    new_img = gain * img ^ gamma
    """
    seg_new = seg
    gain, gamma = (1.2 - 0.8) * np.random.random_sample(2, ) + 0.8
    img_new = np.sign(img) * gain * (np.abs(img) ** gamma)
    return img_new, seg_new


def elastic_deform(img, seg):
    r"""
    Elastic deformation on a image and its target
    """
    img_new, seg_new = elasticdeform.deform_random_grid([img, seg], sigma=2, axis=[(0, 1, 2), (0, 1, 2)], order=[3, 0],
                                                        mode='constant')
    return img_new, seg_new


def combine_augment(img, seg):
    img_new, seg_new = img, seg
    # generate n random decisions for augmentation
    decisions = np.random.randint(2, size=4)
    if np.random.random_sample() > 0.75:
        return img_new, seg_new
    else:
        if decisions[0] == 1:
            img_new, seg_new = flip3d(img_new, seg_new)

        if decisions[1] == 1:
            img_new, seg_new = brightness(img_new, seg_new)

        if decisions[2] == 1:
            img_new, seg_new = rotate3d(img_new, seg_new)

        if decisions[3] == 1:
            img_new, seg_new = elastic_deform(img_new, seg_new)

    return img_new, seg_new
