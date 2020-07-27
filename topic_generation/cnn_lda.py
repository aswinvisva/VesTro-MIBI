import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
import cv2 as cv
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical

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
        self.lda_output = None
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

    def preprocess_features(self, npdata, pca=256):
        """Preprocess an array of features.
        Args:
            npdata (np.array N * ndim): features to preprocess
            pca (int): dim of output
        Returns:
            np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
        """
        _, ndim = npdata.shape
        npdata = npdata.astype('float32')

        npdata = PCA(n_components=100).fit_transform(npdata)

        # L2 normalization
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]

        return npdata

    def _update_labels(self):
        features = Model(self.model.inputs, self.model.get_layer('avg_pool').output).predict(self.X)
        reduced_features = self.preprocess_features(features)

        km = KMeans(n_clusters=self.K,
                    init='k-means++',
                    max_iter=1,
                    random_state=25)
        indices = km.fit_predict(reduced_features)

        # lda_indices = [np.where(i == np.max(i))[0][0] for i in self.lda_output]
        # clustering_loss = adjusted_rand_score(lda_indices, indices)

        image_lists = [[] for _ in range(self.K)]

        for i in range(len(self.X)):
            image_lists[indices[i]].append(i)

        pseudolabels = []
        image_indexes = []

        for cluster, images in enumerate(image_lists):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))

        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}

        new_y = []
        new_X = []

        for j, idx in enumerate(image_indexes):
            pseudolabel = label_to_idx[pseudolabels[j]]
            new_y.append(pseudolabel)
            new_X.append(self.X[idx])

        print(new_y)
        print(image_indexes[0:10])

        categorical_labels = to_categorical(new_y, num_classes=self.K)
        one_hot_vector_labels = np.array(categorical_labels)

        self.X = np.array(new_X)
        self.y = one_hot_vector_labels

    def fit(self, X, bag_of_cells,
            epochs=4,
            batch_size=4):

        # Compute initial labels using LDA
        lda = LDATopicGen(bag_of_cells, topics=self.K)
        y = lda.fit_predict()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        self.X = X_train
        self.y = y_train
        self.lda_output = y_train

        datagen = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # horizontal_flip=True
        )

        datagen.fit(self.X)

        for e in range(epochs):
            print("=" * 5, "Epoch %s" % str(e), "=" * 5)
            batches = 0
            # self._update_labels()

            for x_batch, y_batch in datagen.flow(self.X, self.y, batch_size=batch_size):
                loss = self.model.train_on_batch(x_batch, y_batch)

                batches += 1

                if batches >= len(X_train) / batch_size:
                    print("loss: %s - accuracy: %s" % (loss[0], loss[1]))
                    break

        self.model.evaluate(x=X_test, y=y_test)
