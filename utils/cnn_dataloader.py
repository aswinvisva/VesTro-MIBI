import datetime
from collections import Counter

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import DataLoader, Dataset
import random
import skimage as sk
from skimage import transform
from skimage import util

from feature_extraction.markers_feature_gen import calculate_microenvironment_marker_expression

device = torch.device("cuda")


def random_rotation(image_array, angle=25):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-angle, angle)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]


def preprocess_input(img, n, r_noise=False, r_flip=False, r_rot=False):

    img = np.moveaxis(img, 1, 3)
    img = img.reshape((img.shape[1], img.shape[2], img.shape[3]))

    if r_noise:
        img = random_noise(img)

    if r_flip:
        img = horizontal_flip(img)

    if r_rot:
        img = random_rotation(img)

    img = cv.resize(img, (n, n))

    # if not r_flip and not r_noise and not r_rot:
    #     img = img / 255

    img = np.moveaxis(img, 2, 0)

    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))

    img[np.where(img >= 1)] = 1

    img = torch.tensor(img, dtype=torch.float, device=device)

    return img


class VesselDataset(Dataset):
    def __init__(self,
                 contour_data_multiple_points,
                 markers_data,
                 n=32,
                 batch_size=4,
                 shuffle=True,
                 r_noise=False,
                 r_flip=False,
                 r_rot=False):

        self.contours = np.array([item for sublist in contour_data_multiple_points for item in sublist])
        self.data = np.array(markers_data)
        self.n = n
        self.r_noise = r_noise
        self.r_flip = r_flip
        self.r_rot = r_rot
        self.contours_lookup_dict = {}

        i = 0
        for point_index in range(len(contour_data_multiple_points)):
            for contour_index in range(len(contour_data_multiple_points[point_index])):
                self.contours_lookup_dict[i] = point_index
                i += 1

        print("There are %s samples" % self.contours.shape[0])

        self.iter = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)

    def get_roi(self):
        return self.x.data.cpu().numpy()

    def show_roi(self):
        roi = self.get_roi()
        for i in range(len(roi)):
            cv.imshow("ROI", roi[i])
            cv.waitKey(0)

    def __getitem__(self, index):
        contours = [self.contours[index]]
        marker_data = self.data[self.contours_lookup_dict[index]]
        data, expression_image = calculate_microenvironment_marker_expression(marker_data, contours,
                                                                              plot=False)

        expression_image = preprocess_input(expression_image, self.n)

        return expression_image, expression_image

    def __len__(self):
        return len(self.contours)
