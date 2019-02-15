import numpy as np
from math import sqrt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time


class NN(object):

    def __init__(self, hidden_dims=(350, 700), n_hidden=2, method='Glorot'):
        self.weights = {}
        self.bias = {}
        self.n_hidden = n_hidden
        h1 = hidden_dims[0]
        h2 = hidden_dims[1]
        self.dim_out = 10
        features = 784
        self.dims = [features, h1, h2, self.dim_out]
        self.total_param_nn = features * h1 + h2 * h1 + self.dim_out * h1 + h1 + h2 + self.dim_out
        self.verif_param_nn(self.total_param_nn)
        self.initialize_weights(method)

    def verif_param_nn(self, total_param_nn):
        print('Number of parameter in NN: ', total_param_nn)
        if not (total_param_nn < 1000000 and total_param_nn > 500000):
            raise AssertionError('ERROR! Number of parameter in NN: ' + str(total_param_nn))

    def initialize_weights(self, method='Normal'):
        if method == 'Normal':
            for layer in range(1, len(self.dims)):
                self.weights[layer - 1] = np.random.randn(self.dims[layer], self.dims[layer - 1]) * np.sqrt(
                    2 / self.dims[layer - 1])
                self.bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)
        elif method == 'Zero':
            for layer in range(1, len(self.dims)):
                line = np.zeros((self.dims[layer], self.dims[layer - 1]), dtype=float)
                self.weights[layer - 1] = line
                self.bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)
        else:
            for layer in range(1, len(self.dims)):
                h_layer = self.dims[layer]
                h_last_layer = self.dims[layer - 1]
                d_layer = sqrt(6 / (h_last_layer + h_layer))

                line = np.random.uniform(-d_layer, d_layer, (self.dims[layer], self.dims[layer - 1]))
                self.weights[layer - 1] = line
                self.bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)

    def activation(self, input):
        """
        Relu
        """
        return np.maximum(0., input)

    def softmax(self, input):
        e_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def loss(self, prediction, labels):
        """
        Cross entropy
        """
        return (labels * (-np.log(prediction))).sum()

    def accuracy(self, pred_label, labels):
        return (pred_label == labels).mean()

    def forward(self, input):
        h1_preact = np.dot(input, self.weights[0].T) + self.bias[0]
        h1 = self.activation(h1_preact)
        h2_preact = np.dot(h1, self.weights[1].T) + self.bias[1]
        h2 = self.activation(h2_preact)
        out = self.softmax(np.dot(h2, self.weights[2].T) + self.bias[2])
        predicted_label = out.argmax(axis=1)
        return h1_preact, h1, h2_preact, h2, out, predicted_label

    def backward(self, input, labels, h1_preact, h1, h2_preact, h2, out_softmax, learning_rate):
        batch_size = input.shape[0]

        # Derivative of loss w.r.t softmax
        dl_dsoftmax = out_softmax - labels

        # Gradient of W^(3) and b^(3)
        grad_w3 = np.dot(dl_dsoftmax.T, h2) / batch_size
        grad_b3 = dl_dsoftmax.mean(axis=0)

        # Derivative for hidden 2
        dl_dh2 = np.dot(dl_dsoftmax, self.weights[2])
        dl_dh2_relu = (h2_preact > 0) * dl_dh2

        # Gradient of W^(2) and b^(2)
        grad_w2 = np.dot(dl_dh2_relu.T, h1) / batch_size
        grad_b2 = dl_dh2_relu.mean(axis=0)

        # Derivative for hidden 1
        dl_dh1 = np.dot(dl_dh2, self.weights[1])
        dl_dh1_relu = (h1_preact > 0) * dl_dh1

        # Gradient of W^(1) and b^(1)
        grad_w1 = np.dot(dl_dh1_relu.T, input) / batch_size
        grad_b1 = dl_dh1_relu.mean(axis=0)

        # Update the parameters
        self.update(grad_w3, grad_b3, learning_rate, 2)
        self.update(grad_w2, grad_b2, learning_rate, 1)
        self.update(grad_w1, grad_b1, learning_rate, 0)

    def update(self, grads_w, grads_b, learning_rate, parameter_index):
        self.weights[parameter_index] -= np.multiply(learning_rate, grads_w)
        self.bias[parameter_index] -= np.multiply(learning_rate, grads_b)

    def train(self, input, labels, epochs, learning_rate, batch_size):
        print('Training....')
        start = time.time()
        losses = []
        accuracy = []
        for epoch in range(epochs):
            start_epoch = time.time()
            for i in range(0, input.shape[0], batch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                input_batch = input[i:i + batch_size]
                labels_batch = labels[i:i + batch_size]
                h1_preact, h1, h2_preact, h2, predicted, pred_label = self.forward(input_batch)
                self.backward(input_batch, labels_batch, h1_preact, h1, h2_preact, h2, predicted, learning_rate)
            _, _, _, _, predicted, pred_label = self.forward(input)
            loss = self.loss(predicted, labels)
            losses.append(loss)
            end_epoch = time.time()
            print(f'It took : {(end_epoch - start_epoch):.2f} seconds for epoch {epoch}')
            print(f'Error for epoch {epoch}: {loss:.2f}')
            acc = self.accuracy(pred_label, labels)
            accuracy.append(acc)
            print(f'Model accuracy: {acc}\n')

        plt.plot(losses)
        plt.savefig('losses.png')
        plt.clf()
        plt.plot(accuracy)
        plt.savefig('accuracy.png')
        plt.clf()
        end = time.time()
        print(f'It took : {(end - start)} seconds')
        return np.mean(losses)

    def predict(self, input):
        _, _, _, _, _, pred_label = self.forward(input)
        return pred_label

    def finite_difference(self, labels, grad_test_h2):
        print('test')
        # Const chosen
        N_vector = [10, 500, 1000, 125000, 6250000]
        grads_all = []
        p = min(10, len(self.weights[1]))
        target_weights = self.weights[1][:p]
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
