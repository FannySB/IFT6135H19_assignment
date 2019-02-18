import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def encode_labels(labels):
    labels_reshaped = labels.reshape(len(labels), 1)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    return encoder.fit_transform(labels_reshaped)


def onehot(y, nb_classes=10):
    targets = np.array(y).reshape(-1)
    return np.eye(nb_classes)[targets]


def plot_training_history(history2):
    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history2.history['loss'], 'r', linewidth=2.0)
    plt.plot(history2.history['val_loss'], 'b', linewidth=2.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=20, fontweight='bold')
    plt.savefig("Loss.png")

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history2.history['acc'], 'r', linewidth=2.0)
    plt.plot(history2.history['val_acc'], 'b', linewidth=2.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=20, fontweight='bold')
    plt.savefig("Accuracy.png")
