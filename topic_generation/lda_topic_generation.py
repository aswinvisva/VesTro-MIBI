import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


class LDATopicGen:

    def __init__(self, data, topics=5):
        self.data = data
        self.components = topics
        self.model = None

    def fit_predict(self):
        self.model = LatentDirichletAllocation(n_components=self.components,
                                               random_state=0)

        topics = self.model.fit_transform(self.data)

        print("LDA Perplexity Score %s" % self.model.perplexity(self.data))
        print("LDA Log Likelihood Score %s" % self.model.score(self.data))

        return topics

    def plot(self):
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1.0), "midnightblue"],
                  [norm(-0.5), "seagreen"],
                  [norm(0.5), "mediumspringgreen"],
                  [norm(1.0), "yellow"]]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        ax = sns.clustermap(self.model.components_ / self.model.components_.sum(axis=1)[:, np.newaxis],
                            linewidth=0.5,
                            cmap=cmap
                            )
        plt.show()
