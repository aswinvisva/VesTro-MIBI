from collections import Counter
import argparse

import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def normalize_img(img):
    img = (img - img.mean()) / img.std()

    return img * 255


def mask_pad(img, mask, sigma):
    filterSize = int(2 * np.ceil(2 * sigma) + 1)

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (sigma, sigma))

    dilated_img = cv.dilate(img, element)

    dilated_img[np.isinf(dilated_img)] = 0

    dilated_img = cv.GaussianBlur(dilated_img, (filterSize, filterSize), sigma, borderType=cv.BORDER_CONSTANT)

    colPad = (dilated_img.shape[0] - mask.shape[0]) / 2
    rowPad = (dilated_img.shape[1] - mask.shape[1]) / 2

    padded_img = np.pad(img, ((0, 1), (0, 0)), 'constant')


def imflatfield(I, sigma):
    """Python equivalent imflatfield implementation
       I format must be BGR and type of I must be uint8"""
    A = I.astype(np.float32) / 255  # A = im2single(I);
    Ihsv = cv.cvtColor(A, cv.COLOR_BGR2HSV)  # Ihsv = rgb2hsv(A);
    A = Ihsv[:, :, 2]  # A = Ihsv(:,:,3);

    filterSize = int(2 * np.ceil(2 * sigma) + 1)  # filterSize = 2*ceil(2*sigma)+1;

    # shading = imgaussfilt(A, sigma, 'Padding', 'symmetric', 'FilterSize', filterSize); % Calculate shading
    shading = cv.GaussianBlur(A, (filterSize, filterSize), sigma, borderType=cv.BORDER_REFLECT)

    meanVal = np.mean(A)  # meanVal = mean(A(:),'omitnan')

    # % Limit minimum to 1e-6 instead of testing using isnan and isinf after division.
    shading = np.maximum(shading, 1e-6)  # shading = max(shading, 1e-6);

    B = A * meanVal / shading  # B = A*meanVal./shading;

    # % Put processed V channel back into HSV image, convert to RGB
    Ihsv[:, :, 2] = B  # Ihsv(:,:,3) = B;

    B = cv.cvtColor(Ihsv, cv.COLOR_HSV2BGR)  # B = hsv2rgb(Ihsv);

    B = np.round(np.clip(B * 255, 0, 255)).astype(np.uint8)  # B = im2uint8(B);

    return B, meanVal, shading


def _im_flat_field_correction(img: np.ndarray,
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

    return img.astype(np.float32), mean_val.astype(np.float32), shading.astype(np.float32)


def stitched_data_denoise(bg: np.ndarray,
                          img: np.ndarray,
                          filter_size: tuple = (3, 3),
                          sigma: int = 50):
    cap = 100
    floor = 0

    raw_mask_data_cap = bg

    raw_mask_data_cap[raw_mask_data_cap < floor] = 0
    raw_mask_data_cap[raw_mask_data_cap > cap] = cap

    raw_mask_data_g = cv.GaussianBlur(raw_mask_data_cap, (filter_size, filter_size), sigma,
                                      borderType=cv.BORDER_REFLECT)

    mask = cv.adaptiveThreshold(raw_mask_data_g, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 11, 2)

    cv.imwrite("before.png", img)

    bg_flat_field_corrected, bg_mean_val, bg_shading = imflatfield(bg, sigma)

    img, mean_val, shading = imflatfield(img, sigma)

    before_img = (img - img.mean()) / img.std()

    plt.imshow(before_img, cmap="Greys", vmin=np.min(before_img), vmax=np.max(before_img))
    cv.imwrite("before_flat_field_corrected.png", before_img * 255)

    shading_norm = shading / np.max(shading)
    bg_shading_norm = shading / np.max(bg_shading)

    mean_shading = np.mean(np.concatenate((shading_norm, bg_shading_norm), axis=0))

    out = img / mean_shading

    # after_img = normalize_img(out)

    cv.imwrite("after.png", out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument('data_path', type=str,
                        help='Path to TIFF file')

    parser.add_argument('bg_path', type=str,
                        help='Path to TIFF file')

    # Optional positional argument
    parser.add_argument('--filter_radius', type=int, default=3,
                        help='Filter Radius')

    # Optional argument
    parser.add_argument('--sigma', type=int, default=10,
                        help='Standard Deviation of Gaussian filter')

    args = parser.parse_args()

    bg = np.array(Image.open(args.bg_path).convert("RGB"))
    data = np.array(Image.open(args.data_path).convert("RGB"))

    crop_left = 500
    crop_right = 1250
    crop_top = 1650
    crop_bottom = 500

    data = data[crop_left:data.shape[0] - crop_right, crop_bottom:data.shape[1] - crop_top]

    bg = normalize_img(bg)
    data = normalize_img(data)

    #
    # h = 512
    # w = int(data.shape[1] * (512/data.shape[0]))
    #
    # small = cv.resize(data, (w, h))
    #
    # cv.imshow("ASDA", small)
    # cv.waitKey(0)

    stitched_data_denoise(bg, data,
                          filter_size=(args.filter_radius, args.filter_radius),
                          sigma=args.sigma)
