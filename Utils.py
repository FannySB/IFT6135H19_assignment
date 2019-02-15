import numpy as np
from sklearn.preprocessing import OneHotEncoder


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def encode_labels(labels):
    labels_reshaped = labels.reshape(len(labels), 1)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    return encoder.fit_transform(labels_reshaped)
