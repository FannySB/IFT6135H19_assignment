import numpy as np
from math import sqrt
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import _pickle as pickle
import gzip


class NN(object):

    def __init__(self, hidden_dims=(8, 10), n_hidden=2, mode='train', datapath=None, model_path=None, learning_rate=0.01):
        print('Init')
        self.learning_rate = learning_rate
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
        if not (total_param_nn < 1000000 and total_param_nn > 500000):
            raise AssertionError('ERROR! Number of parameter in NN: ' + str(total_param_nn))

    def initialize_weights(self, method='Normal'):
        print('initialize_weights')
        weights = {}
        bias = {}

        if method == 'Normal':
            for layer in range(1, len(self.dims)):
                weights[layer - 1] = np.random.randn(self.dims[layer - 1], self.dims[layer]) * np.sqrt(2 / self.dims[layer - 1])
                bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)
        elif method == 'Zero':
            for layer in range(1, len(self.dims)):
                line = np.zeros((self.dims[layer -1], self.dims[layer]), dtype=float)
                weights[layer - 1] = line
                bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)
        else:
            for layer in range(1, len(self.dims)):
                h_layer = self.dims[layer]
                h_last_layer = self.dims[layer - 1]
                d_layer = sqrt(6/ (h_last_layer + h_layer))

                line = np.random.uniform(-d_layer, d_layer)
                weights[layer - 1] = line
                bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)

        return weights, bias

    def forward(self, input, weights, bias):
        h1 = self.activation(np.dot(input, weights[0]) + bias[0])
        h2 = self.activation(np.dot(h1, weights[1]) + bias[1])
        out = self.softmax(np.dot(h2, weights[2]) + bias[2])
        return out

    def activation(self, input):
        """
        Relu
        :param input:
        :return:
        """
        return np.maximum(0, input)

    def loss(self, prediction, labels):
        """
        Cross entropy
        :param prediction:
        :param labels:
        :return:
        """
        # print('loss')
        pred = np.max(prediction, axis=0)
        return -np.sum(labels * np.log(pred))

    def softmax(self, input):
        # print('softmax')
        input -= np.max(input)
        #print('Values softmax:', np.exp(input) / np.sum(np.exp(input)))
        return np.exp(input) / np.sum(np.exp(input))

    def backward(self, cache, labels, input, weights, bias):
        print('backward')
        # Relu derivative
        relu_d =  1 * (input > 0)

        length = input.shape[0]
        # SGD

    def update(self, grads_w, grads_b, weight, bias):
        """
        SDG
        :param grads:
        :return:
        """
        weight -= np.multiply(learning_rate, grads_w)
        bias -= np.multiply(learning_rate, grads_b)

        return weight, bias

        print('update')

    def train(self, input, labels):
        # print('train')
        # predicted = self.forward()
        # self.loss(predicted, labels)
        return input

    def test(self, weights, prediction, label):
        print('test')
        N_vector = [1, 10, 1000, 125000, 6250000]
        for N in N_vector:
            epsilon = 1/N
            grads = []
            p = min(10, len(weights))
            target_weights = weights[:p]
            target_labels = labels[:p]
            for i in range(p):
                mask = np.zeros(p)
                mask[i] = epsilon
                grads.append((mlp.loss(np.sum(target_weights, mask)) - mlp.loss(np.diff(weights, mask)))/ 2*epsilon)

            plt.plot(grads)
            plt.savefig('test_grads' + str(N) + '.png')



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
# weights, bias = mlp.initialize_weights(method='Zero')
weights, bias = mlp.initialize_weights(method='Normal')
# weights, bias = mlp.initialize_weights(method='Glorot')
epochs = 10
learning_rate = 0.01
mean_error = []
for epoch in range(epochs):
    cum_error = 0
    for idx in range(len(train_sample)):
        data = train_sample[idx]
        labels = train_labels_sample[idx]

        predicted = mlp.forward(data, weights, bias)
        error = mlp.loss(predicted, labels)

        #backprop missing
        # weights, bias = mlp.update(grads_w, grads_b, weights, bias)

        cum_error += error
        # print("predicted = ", predicted)
        # print("loss ", idx_loss)
    mean_error.append(cum_error/len(train_sample))
    print('epoch:', epoch, 'mean_error:', mean_error[epoch])
plt.plot(mean_error)
plt.savefig('mean_error.png')

# Test on the 2nd hidden layer by homework's instructions
predictions = mlp.forward(data, weights, bias)
mlp.test(weights[1], predictions[0], labels[0])

