import numpy as np
from math import sqrt
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import _pickle as pickle
import gzip
import time


class NN(object):

    def __init__(self, hidden_dims=(350, 700), n_hidden=2, mode='train', datapath=None, model_path=None,
                 learning_rate=0.001):
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        h1 = hidden_dims[0]
        h2 = hidden_dims[1]
        self.dim_out = 10
        features = 784
        self.dims = [features, h1, h2, self.dim_out]
        self.total_param_nn = features * h1 + h2 * h1 + self.dim_out * h1 + h1 + h2 + self.dim_out

    def verif_param_nn(self, total_param_nn):
        print('Number of parameter in NN: ', total_param_nn)
        if not (total_param_nn < 1000000 and total_param_nn > 500000):
            raise AssertionError('ERROR! Number of parameter in NN: ' + str(total_param_nn))

    def initialize_weights(self, method='Normal'):
        weights = {}
        bias = {}

        if method == 'Normal':
            for layer in range(1, len(self.dims)):
                weights[layer - 1] = np.random.randn(self.dims[layer - 1], self.dims[layer]) * np.sqrt(
                    2 / self.dims[layer - 1])
                bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)
        elif method == 'Zero':
            for layer in range(1, len(self.dims)):
                line = np.zeros((self.dims[layer - 1], self.dims[layer]), dtype=float)
                weights[layer - 1] = line
                bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)
        else:
            for layer in range(1, len(self.dims)):
                h_layer = self.dims[layer]
                h_last_layer = self.dims[layer - 1]
                d_layer = sqrt(6 / (h_last_layer + h_layer))

                line = np.random.uniform(-d_layer, d_layer, (self.dims[layer - 1], self.dims[layer]))
                weights[layer - 1] = line
                bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)

        return weights, bias

    def activation(self, input):
        """
        Relu
        :param input:
        :return:
        """
        return np.maximum(0., input)

    def softmax(self, input):
        e_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, input, weights, bias):
        h1_preact = np.dot(input, weights[0]) + bias[0]
        h1 = self.activation(h1_preact)
        h2_preact = np.dot(h1, weights[1]) + bias[1]
        h2 = self.activation(h2_preact)
        out = self.softmax(np.dot(h2, weights[2]) + bias[2])
        max = np.argmax(out, axis=1)
        pred_label = np.zeros((len(input), self.dim_out))
        for i, j in enumerate(max):
            pred_label[i][j] = 1
        return h1_preact, h1, h2_preact, h2, out, pred_label

    def loss(self, prediction, labels):
        """
        Cross entropy
        :param prediction:
        :param labels:
        :return:
        """
        pred = []
        for i in range(len(labels)):
            pred.append(-np.log(prediction[i][np.argmax(labels[i])]))
        return np.sum(pred)

    def accuracy(self, pred_label, labels):
        return accuracy_score(labels, pred_label)

    def backward(self, input, labels, h1_preact, h1, h2_preact, h2, out, weights, bias, learning_rate):
        norm_grad = []
        grad_test_h2 = []
        for sgd_index in range(input.shape[0]):
            # Derivative of loss w.r.t softmax
            dl_dsoftmax = np.add(out[sgd_index, :], - labels[sgd_index, :])

            # Gradient of W^(3) and b^(3)
            grad_w3 = np.outer(h2[sgd_index, :], dl_dsoftmax)
            grad_b3 = dl_dsoftmax

            # Derivative for hidden 2
            dl_dh2 = np.dot(weights[2], dl_dsoftmax)
            dl_dh2_relu = (h2_preact[sgd_index, :] > 0) * dl_dh2

            # Gradient of W^(2) and b^(2)
            grad_w2 = np.outer(h1[sgd_index, :], dl_dh2_relu)
            grad_b2 = dl_dh2_relu
            grad_test_h2 = grad_w2

            # Test for gradient
            norm_test = np.absolute(grad_test_h2)
            norm_grad.append(np.max(norm_test))

            # Derivative for hidden 1
            dl_dh1 = np.dot(weights[1], dl_dh2)
            dl_dh1_relu = (h1_preact[sgd_index, :] > 0) * dl_dh1

            # Gradient of W^(1) and b^(1)
            grad_w1 = np.outer( input[sgd_index, :], dl_dh1_relu)
            grad_b1 = dl_dh1_relu

            # Update the parameters
            weights[2], bias[2] = self.update(grad_w3, grad_b3, weights[2], bias[2], learning_rate)
            weights[1], bias[1] = self.update(grad_w2, grad_b2, weights[1], bias[1], learning_rate)
            weights[0], bias[0] = self.update(grad_w1, grad_b1, weights[0], bias[0], learning_rate)

        plt.plot(norm_grad)
        plt.savefig('Norm_grad.png')
        plt.clf()
        return weights, bias, grad_test_h2

    def mini_batch(self, input, labels, h1_preact, h1, h2_preact, h2, out, weights, bias, learning_rate, batch_size):
        input, labels = shuffle(input, labels)
        for i in range(0, input.shape[0], batch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            input_mini = input[i:i + batch_size]
            labels_mini = labels[i:i + batch_size]
            weights, bias, grad_test_h2 = self.backward(input_mini, labels_mini, h1_preact, h1, h2_preact, h2, out,
                                                        weights, bias, learning_rate)
        return weights, bias, grad_test_h2

    def update(self, grads_w, grads_b, weight, bias, learning_rate):
        weight -= np.multiply(learning_rate, grads_w)
        bias -= np.multiply(learning_rate, grads_b)
        return weight, bias

    def train(self, input, labels, weights, bias, epochs, learning_rate, batch_size):
        print('Training....')
        start = time.time()
        errors = []
        for epoch in range(epochs):
            start_epoch = time.time()
            h1_preact, h1, h2_preact, h2, predicted, pred_label = self.forward(input, weights, bias)
            acc = self.accuracy(pred_label, labels)
            print(f'Model accuracy: {acc}\n')
            error = self.loss(predicted, labels)
            errors.append(error)
            weights, bias, grad_test_h2 = self.mini_batch(input, labels, h1_preact, h1, h2_preact, h2, predicted,
                                                          weights, bias, learning_rate, batch_size)
            end_epoch = time.time()
            print(f'It took : {(end_epoch - start_epoch):.2f} seconds for epoch {epoch}')
            print(f'Error for epoch {epoch}: {error:.2f}')

        plt.plot(errors)
        plt.savefig('errors.png')
        plt.clf()
        end = time.time()
        print(f'It took : {(end - start)} seconds')
        return weights, bias, grad_test_h2

    def test(self, weights, labels, grad_test_h2):
        print('test')
        # Const chosen TODO
        N_vector = [10, 500, 1000, 125000, 6250000]
        grads_all = []
        p = min(10, len(weights))
        target_weights = weights[:p]
        target_weights = self.softmax(target_weights)
        for N in N_vector:
            epsilon = 1 / N
            grads = []
            for i in range(p):
                mask = np.zeros(p)
                mask[i] = epsilon
                eps_added = np.add(np.transpose(target_weights), mask)
                mask[i] = - epsilon
                eps_subs = np.add(np.transpose(target_weights), mask)
                grad = (self.loss(eps_added, labels) - self.loss(eps_subs, labels)) / 2 * epsilon
                grads.append(grad)
            grads_all.append(grads)
        temp = grad_test_h2[1][:p]
        temp2 = [-x for x in temp]
        diff_grads = np.add(grads_all, temp2)
        plt.plot(diff_grads)
        plt.savefig('grads_test.png')
        plt.clf()


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def encode_labels(labels):
    labels_reshaped = labels.reshape(len(labels), 1)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    return encoder.fit_transform(labels_reshaped)


mnist = np.load('datasets/mnist.pkl.npy')
train = mnist[0, 0]
train_size = 1000
train_norm = normalize(train, axis=0)
train_sample = train_norm[:train_size]
train_labels = mnist[0, 1]
train_labels_sample = encode_labels(train_labels[:train_size])
validation = mnist[1, 0]
validation_labels = mnist[1, 1]
test = mnist[2, 0]
test_labels = mnist[2, 1]

batch_size = 20
epochs = 10
learning_rate = 10 ** (-2)
mlp = NN(hidden_dims=(100, 200))
# mlp.verif_param_nn(mlp.total_param_nn)
# weights, bias = mlp.initialize_weights(method='Zero')
weights, bias = mlp.initialize_weights(method='Glorot')
# weights, bias = mlp.initialize_weights(method='Normal')
weights, bias, grad_test_h2 = mlp.train(train_sample, train_labels_sample, weights, bias, epochs, learning_rate,
                                        batch_size)
_, _, _, _, predicted, chosen_class = mlp.forward(train_sample, weights, bias)
acc = mlp.accuracy(chosen_class, train_labels_sample)
print('Model accuracy: ', acc)

mlp.test(weights[1], train_labels_sample[-1], grad_test_h2)
