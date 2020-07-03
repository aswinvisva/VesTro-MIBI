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

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''

class CNNFeatureGen:

    def __init__(self, n=128):
        xception_base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(n, n, 3), pooling='avg')
        self.n = n

        for layer in xception_base_model.layers:
            layer.trainable = False

        x = Flatten()(xception_base_model.output)

        self.model = Model(inputs=xception_base_model.input, outputs=x)
        self.model.summary()

    def generate(self, img):
        img = cv.resize(img, (self.n, self.n))
        # # img = preprocess_input(img)
        # cv.imshow("Keypoint", img)
        # cv.waitKey(0)

        img = img.reshape(1, self.n, self.n, 3)
        feature_vec = self.model.predict(img)

        return feature_vec
