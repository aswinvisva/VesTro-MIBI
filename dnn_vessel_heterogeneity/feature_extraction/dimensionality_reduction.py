import numpy as np
from sklearn.decomposition import PCA
import SimpSOM as sps

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''

def pca_reduction(data,
                  vec_size=32):
    pca = PCA(n_components=vec_size)
    pca.fit(data)
    data = pca.transform(data)

    return data


def som_reduction(data):
    net = sps.somNet(20, 20, data, PBC=True)
    net.train(0.01, 1000)
    net.save('som_weights')
    net.nodes_graph(colnum=0)
    net.diff_graph()
    clusters = net.cluster(data, type='qthresh')
    print(clusters)

    return clusters[0]

