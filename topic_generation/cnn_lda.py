import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
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
        self.X = None
        self.y = None

        xception_base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(self.n, self.n, 3),
                                       pooling='avg')
        flatten = Flatten()(xception_base_model.output)
        classifier = Dense(self.K, activation="softmax")(flatten)

        # for layer in xception_base_model.layers:
        #     layer.trainable = False

        self.model = Model(inputs=xception_base_model.input, outputs=classifier)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def fit(self, X, bag_of_cells,
            epochs=10,
            batch_size=16):

        # Compute initial labels using LDA
        lda = LDATopicGen(bag_of_cells, topics=self.K)
        y = lda.fit_predict()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        self.X = X_train
        self.y = y_train

        datagen = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # horizontal_flip=True
        )

        datagen.fit(X_train)

        for e in range(epochs):
            print("="*5, "Epoch %s" % str(e), "="*5)
            batches = 0
            for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
                loss = self.model.train_on_batch(x_batch, y_batch)

                batches += 1

                if batches >= len(X_train) / batch_size:
                    print("loss: %s - accuracy: %s" % (loss[0], loss[1]))
                    break

        self.model.evaluate(x=X_test, y=y_test)

        for x in X_test:
            topic = self.model.predict(np.array([x]))
            print(topic)
            cv.imshow("ASD", x)
            cv.waitKey(0)
