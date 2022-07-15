import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from skimage import transform as skimage_transform


def getAttMap(img, attMap, blur=True, overlap=True, norm=True):
    if norm:
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
    if blur:
        attMap = scipy.ndimage.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + (attMap ** 0.7).reshape(
            attMap.shape + (1,)) * attMapV
    return attMap


def visualize(image, mask, norm=False, title=None, show=True):
    image = np.float32(image) / 255
    gradcam_image = getAttMap(image, mask, overlap=True, blur=True, norm=norm)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("Image")
        ax.imshow(gradcam_image)
        if title:
            plt.title(title)
        plt.show()
    return gradcam_image


def plt_grid(images, label=None, title=None, save=False, scale=4):
    x = math.floor(math.sqrt(len(images)))
    y = math.ceil(len(images) / x)

    h, w, _ = images[0].shape

    fig, ax = plt.subplots(x, y, figsize=(math.floor(scale * w / h) * x, math.floor(scale * h / w) * y))
    ax = ax.flatten()

    if title:
        fig.suptitle(title)

    for i, image in enumerate(images):
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        if label is not None:
            ax[i].set_xlabel(label[i])
        ax[i].imshow(image)

    for j in range(len(images), x * y):
        fig.delaxes(ax[j])

    fig.tight_layout()

    if save:
        plt.savefig(save, dpi=200)

    plt.show()
