import hdbscan
from sklearn.decomposition import *
from sklearn.manifold import *
from minisom import MiniSom
import numpy as np
from umap import UMAP


def umap(X, **kwargs):
    n_components = kwargs.get("n_components", 2)
    random_state = kwargs.get("random_state", 42)
    n_neighbors = kwargs.get("n_neighbors", 15)

    return UMAP(n_components=n_components, random_state=random_state, n_neighbors=n_neighbors).fit_transform(X)


def pca(X, **kwargs):
    n_components = kwargs.get("n_components", 2)
    random_state = kwargs.get("random_state", 42)

    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)


def tsne(X, **kwargs):
    n_components = kwargs.get("n_components", 2)
    random_state = kwargs.get("random_state", 42)

    return TSNE(n_components=n_components, random_state=random_state).fit_transform(X)


def svd(X, **kwargs):
    n_components = kwargs.get("n_components", 2)
    random_state = kwargs.get("random_state", 42)

    return TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(X)


def som(X, **kwargs):
    size = kwargs.get("size", 50)
    epochs = kwargs.get("epochs", 10000)
    random_state = kwargs.get("random_state", 42)

    som = MiniSom(size, size, len(X[0]),
                  neighborhood_function='gaussian', sigma=1.5,
                  random_seed=random_state)

    som.pca_weights_init(X)
    som.train_random(X, epochs, verbose=True)
