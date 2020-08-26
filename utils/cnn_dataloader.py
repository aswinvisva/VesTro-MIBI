import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
import cv2 as cv

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class VesselDataset:
    def __init__(self, contours, segmentation_mask):
        self.x = []

        for i in range(len(contours)):
            for cnt in contours[i]:
                x, y, w, h = cv.boundingRect(cnt)
                roi = segmentation_mask[i][y:y + h, x:x + w]
                self.x.append(roi)

        self.x = np.array(self.x)

    def get_roi(self):
        return self.x

    def show_roi(self):
        for roi in self.get_roi():
            cv.imshow("ROI", roi)
            cv.waitKey(0)
