from collections import Counter
import argparse

import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def im_flat_field_correction(img: np.ndarray,
                             filter_size: tuple = (51, 51),
                             sigma: int = 50):
    """
    Get vessel boundaries beyond which a vessel cannot expand

    :param img: np.ndarray, MxNxK image to be corrected (If 3-channel, must be BGR)
    :param filter_size: tuple, Size of gaussian filter
    :param sigma: int, Sigma

    :return: list, [n_vessels, point_size[0], point_size[1]] of region masks for each vessel beyond which it cannot
    expand
    """

    img = img.astype(np.float32)
    plt.imshow(img * 100, cmap="Greys")
    plt.show()

    m, n, k = img.shape

    if k == 3:
        # 3-Channel Image

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)

        img = hsv[:, :, 2]

    shading = cv.GaussianBlur(img, filter_size, sigma)

    mean_val = np.nanmean(img)

    B = img * mean_val / shading

    B = np.nan_to_num(B)

    B[np.isinf(B)] = 0.0

    if k == 3:
        hsv = hsv.astype(np.float32)
        B = B.astype(np.float32)

        hsv[:, :, 2] = B

        img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img*100, cmap="Greys")
    plt.show()

    return B


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument('path', type=str,
                        help='Path to TIFF file')

    # Optional positional argument
    parser.add_argument('--filter_radius', type=int,
                        help='Filter Radius')

    # Optional argument
    parser.add_argument('--sigma', type=int,
                        help='Standard Deviation of Gaussian filter')

    args = parser.parse_args()

    segmentation_mask = np.array(Image.open(args.path).convert("RGB"))

    im_flat_field_correction(segmentation_mask,
                             filter_size=(args.filter_radius, args.filter_radius),
                             sigma=args.sigma)
