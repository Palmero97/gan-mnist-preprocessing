# -*- coding: utf-8 -*-
"""Transformer classes

This module contains the following data transformers:
    - IdTransfomer
    - InvColorTransformer
    - PCATranfomer
    - NoisyPcaTransformer
"""

import numpy as np
from sklearn.decomposition import PCA


class IdTransformer:
    """
    Id transformer.
    """

    def transform(self, x):
        return x


class InvColorTransformer:
    """
    Inverse color of greyscale img transformer.
    """

    def transform(self, x):
        return 255 - x


class ImgPcaTransformer:
    """
    Applies PCA decomposition and projects the input instances into the
    reduced feature space. It is adapted for square images.
    """

    def __init__(self, h):
        self._h = h
        self.n_components = h*h
        self._pca = PCA(self.n_components)
        self._fitted = False

    def fit(self, x):
        self._pca.fit(x.reshape(x.shape[0], -1))

        print("Explained variation: {}".
              format(sum(self._pca.explained_variance_ratio_)))
        print("Number of components: {}".
              format(self._pca.n_components_))

        return self

    def transform(self, x):
        if not self._fitted:
            self.fit(x)
            self._fitted = True

        x_pca = self._pca.transform(x.reshape(x.shape[0], -1))

        # Reshape to appropiate image shape
        x_pca = x_pca.reshape((x.shape[0], self._h, self._h))

        return x_pca


class ProjectedPcaTransformer:
    """
    Applies PCA decomposition, projects the input instances into the
    reduced feature space, and then they are projected back to the original
    space.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self._pca = PCA(n_components)
        self._fitted = False

    def fit(self, x):
        self._pca.fit(x.reshape(x.shape[0], -1))

        print("Explained variation: {}".
              format(sum(self._pca.explained_variance_ratio_)))
        print("Number of components: {}".
              format(self._pca.n_components_))

        return self

    def transform(self, x):
        if not self._fitted:
            self.fit(x)
            self._fitted = True

        x_pca = self._pca.transform(x.reshape(x.shape[0], -1))

        # Reshape to the original shape
        x_projected = self._pca.inverse_transform(x_pca).reshape(x.shape)

        return x_projected


class NoisyProjectedPcaTransformer(ProjectedPcaTransformer):
    """
    Applies PCA transformer, projects the input instances into the
    reduced feature space, then they are projected into the original space
    and then adds gaussian noise to those projected instances.
    """

    def __init__(self, n_components, noise_factor):
        ProjectedPcaTransformer.__init__(self, n_components)
        self.noise_factor = noise_factor

    def transform(self, x):
        x_projected = super().transform(x)

        # Add gaussian noise
        noise = np.random.normal(0, 1, x_projected.shape) * self.noise_factor

        return x_projected + noise
