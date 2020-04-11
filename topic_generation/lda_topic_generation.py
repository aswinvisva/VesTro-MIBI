import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

class LDATopicGen:

    def __init__(self, data):
        self.data = data
        self.components = 100
        self.model = None

    def fit_model(self):
        '''
        TODO: Fit the model to the encoded features
        '''

        self.model = LatentDirichletAllocation(n_components=self.components,
                                               random_state=0)

        topics = self.model.fit_transform(self.data)

        return topics
