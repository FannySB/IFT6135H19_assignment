import numpy as np
from sklearn.preprocessing import OneHotEncoder
import _pickle as pickle
import gzip


class NN(object):

    def __init__(self, hidden_dims=(1024, 2048), n_hidden=2, mode='train', datapath=None, model_path=None):
        print('Init')
        self.n_hidden = n_hidden
        h1 = hidden_dims[0]
        h2 = hidden_dims[1]
        out = 10
        features = 784
        self.dims = [features, h1, h2, out]
        total_param_h1 = features + h1 * features + h1 * 1
        total_param_h2 = total_param_h1 + h2 * total_param_h1 + h2 * 1
        self.total_param_nn = total_param_h2 + out * total_param_h2

    def verif_param_nn(self, total_param_nn):
        print('verif_param_nn')
        assert (not (total_param_nn < 1000000 & total_param_nn > 500000),
                'ERROR! Number of parameter in NN: ' + str(total_param_nn))

    # %%
    def initialize_weights(self, method='Normal'):
        print('initialize_weights')
        weights = {}
        bias = {}
        if method == 'Normal':
            for layer in range(len(self.dims) - 1):
                weights[layer] = np.random.normal(0, 0.05, size=(self.dims[layer], self.dims[layer + 1]))
                bias[layer] = np.random.normal(0, 0.05, size=self.dims[layer + 1])
        elif method == 'Zero':
            for layer in range(len(self.dims) - 1):
                line = np.zeros((self.dims[layer], self.dims[layer + 1]), dtype=float)
                weights[layer] = line
                bias[layer] = np.zeros((self.dims[layer + 1],), dtype=float)
        else:
            for layer in range(len(self.dims) - 1):
                weights[layer] = np.random.rand(self.dims[layer], self.dims[layer + 1])
                bias[layer] = np.random.rand(self.dims[layer + 1])
        return weights, bias

    # %%
    def forward(self, input, weights, bias):
        print('forward')
        h1 = self.activation(np.dot(input, weights[0]) + bias[0])
        h2 = self.activation(np.dot(h1, weights[1]) + bias[1])
        out = self.softmax(np.dot(h2, weights[2]) + bias[2])
        print(h1.shape, h2.shape, out.shape)
        return out

    # %%
    def activation(self, input):
        return np.maximum(0, input)

    # %%
    def loss(self, prediction, labels):
        print('loss')
        pred = np.max(prediction, axis=0)
        return -np.sum(labels * np.log(pred))

    # %%
    def softmax(self, input):
        print('softmax')
        input -= np.max(input)
        #print('Values softmax:', np.exp(input) / np.sum(np.exp(input)))
        return np.exp(input) / np.sum(np.exp(input))

    # %%
    def backward(self, cache, labels):
        print('backward')
        # SGD

    # %%
    def update(self, grads):
        print('update')

    # %%
    def train(self, input, labels):
        # print('train')
        # predicted = self.forward()
        # self.loss(predicted, labels)
        return input

    # %%
    def test(self):
        print('test')


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def encode_labels(labels):
    labels_reshaped = labels.reshape(len(labels), 1)
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(labels_reshaped)


mnist = np.load('datasets/mnist.pkl.npy')
train = mnist[0, 0]
# train_norm = normalize(train, axis=0)
train_norm = train
train_sample = train_norm[:100]
train_labels = mnist[0, 1]
train_labels_sample = encode_labels(train_labels[:100])
validation = mnist[1, 0]
validation_labels = mnist[1, 1]
test = mnist[2, 0]
test_labels = mnist[2, 1]

mlp = NN()
mlp.verif_param_nn(mlp.total_param_nn)
weights, bias = mlp.initialize_weights(method='Normal')
predicted = mlp.forward(train_sample, weights, bias)
loss = mlp.loss(predicted, train_labels_sample)


# epochs = 1
# learning_rate = 0.01
# for epoch in range(epochs):
#     cum_loss = 0
#     for idx in range(len(train_sample)):
#         data = train_sample[idx]
#         labels = train_labels_sample[idx]
#
#         predicted = mlp.forward(data, weights, bias)
#         ce = mlp.loss(predicted, labels)
#
#         print("predicted = ", predicted)
#         print("loss ", ce )
#         print(ce)
