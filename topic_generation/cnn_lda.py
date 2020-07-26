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
from sklearn.model_selection import train_test_split

from topic_generation.lda_topic_generation import LDATopicGen

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


class CNNLDA:

    def __init__(self, K=16, n=128):
        self.n = n
        self.K = K

        xception_base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(self.n, self.n, 3), pooling='avg')
        flatten = Flatten()(xception_base_model.output)
        classifier = Dense(self.K, activation="softmax")(flatten)

        # for layer in xception_base_model.layers:
        #     layer.trainable = False

        self.model = Model(inputs=xception_base_model.input, outputs=classifier)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def fit(self, X, bag_of_cells):
        lda = LDATopicGen(bag_of_cells, topics=self.K)
        y = lda.fit_predict()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        self.model.fit(X_train, y_train, epochs=10)
        self.model.evaluate(x=X_test, y=y_test)
