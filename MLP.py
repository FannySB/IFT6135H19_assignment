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
        # total_param_h1 = features + h1 * features + h1 * 1
        # total_param_h2 = total_param_h1 + h2 * total_param_h1 + h2 * 1
        # self.total_param_nn = total_param_h2 + self.dim_out * total_param_h2
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

                line = np.random.uniform(-d_layer, d_layer)
                weights[layer - 1] = line
                bias[layer - 1] = np.zeros((self.dims[layer],), dtype=float)

        return weights, bias

    def activation(self, input):
        """
        Relu
        :param input:
        :return:
        """
        return np.maximum(0, input)

    def softmax(self, input):
        input -= np.max(input)
        return np.exp(input) / np.sum(np.exp(input))

    def forward(self, input, weights, bias):
        h1 = self.activation(np.dot(input, weights[0]) + bias[0])
        h2 = self.activation(np.dot(h1, weights[1]) + bias[1])
        out = self.softmax(np.dot(h2, weights[2]) + bias[2])
        pred_label_bin = LabelBinarizer()
        pred_label = pred_label_bin.fit_transform(out)
        return h1, h2, out, pred_label

    def loss(self, prediction, labels):
        """
        Cross entropy
        :param prediction:
        :param labels:
        :return:
        """
        pred = np.max(prediction, axis=0)
        return -np.sum(labels * np.log(pred))

        grads = []
        for sgd_index in range(input.shape[0]):
            # Derivative of loss w.r.t softmax
            dl_dsoftmax = np.dot(labels[sgd_index, :], out[sgd_index, :] - np.identity(self.dim_out))

            # Gradient of W^(3) and b^(3)
            grad_w3 = np.outer(h2[sgd_index, :], dl_dsoftmax)
            grad_b3 = dl_dsoftmax

            # Derivative for hidden 2
            dl_dh2 = np.dot(weights[2], dl_dsoftmax)
            dl_dh2_relu = (h2[sgd_index, :] > 0) * dl_dh2

            # Gradient of W^(2) and b^(2)
            grad_w2 = np.outer(h1[sgd_index, :], dl_dh2_relu)
            grad_b2 = dl_dh2_relu

            # Derivative for hidden 1
            dl_dh1 = np.dot(weights[1], dl_dh2)
            dl_dh1_relu = (h1[sgd_index, :] > 0) * dl_dh1

            # Gradient of W^(1) and b^(1)
            grad_w1 = np.outer(input[sgd_index, :], dl_dh1_relu)
            grad_b1 = dl_dh1_relu
            grads.append(grad_b1)

            # Update the parameters
            weights[2], bias[2] = self.update(grad_w3, grad_b3, weights[2], bias[2], learning_rate)
            weights[1], bias[1] = self.update(grad_w2, grad_b2, weights[1], bias[1], learning_rate)
            weights[0], bias[0] = self.update(grad_w1, grad_b1, weights[0], bias[0], learning_rate)

        # print('grads0', grads[0].shape)
        plt.plot(grads[0])
        plt.savefig('grads.png')
        plt.clf()

        return weights, bias

    def mini_batch(self, input, labels, h1, h2, out, weights, bias, learning_rate, batch_size):
        input, labels = shuffle(input, labels)

        for i in range(0, input.shape[0], batch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            input_mini = input[i:i + batch_size]
            labels_mini = labels[i:i + batch_size]
            weights, bias = self.backward(input_mini, labels_mini, h1, h2, out, weights, bias, learning_rate)

        return weights, bias

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
            h1, h2, predicted, pred_label = mlp.forward(input, weights, bias)
            acc = self.accuracy(pred_label, labels)
            print('Model accuracy: ', acc)
            error = mlp.loss(predicted, labels)
            weights, bias = mlp.mini_batch(input, labels, h1, h2, predicted, weights, bias, learning_rate, batch_size)
            end_epoch = time.time()
            print(f'It took : {(end_epoch - start_epoch):.2f} seconds for epoch {epoch}')
            print(f'Error for epoch {epoch}: {error:.2f}\n')

        mlp.test(weights[1], predicted[0], labels[0])
        plt.plot(errors)
        plt.savefig('errors.png')
        end = time.time()
        print(f'It took : {(end - start)} seconds')

    def test(self, weights, prediction, labels):
        print('test')
        #Const chosen TODO
        N_vector = [10, 50, 1000, 125000, 6250000]
        i_vector = [1, 1, 3, 3, 4]
        grads_all = []
        for cpt_n in range(len(N_vector)):
            epsilon =  1 / N_vector[cpt_n]
            grads = []
            p = min(10, len(weights))
            target_weights = weights[:p]
            target_labels = labels[:p]

            mask = np.zeros(p)
            mask[i_vector[cpt_n]] = epsilon
            print('mask', mask)
            print('np.transpose(target_weights)', np.transpose(target_weights).shape)
            eps_added = np.add(np.transpose(target_weights), mask)
            mask[i_vector[cpt_n]] = - epsilon
            eps_subs = np.add(np.transpose(target_weights), mask)
            print('loss+', self.loss(eps_added, labels))
            print('loss-', self.loss(eps_subs, labels))
            print('loss-loss', self.loss(eps_added, labels) - self.loss(eps_subs, labels))
            grad = (self.loss(eps_added, labels) - self.loss(eps_subs, labels)) / 2 * epsilon
            grads.append(grad)
            print('grad', grad)

        print(len(grads), grads)
        plt.plot(grads)
        plt.savefig('grads_test.png')
        plt.clf()
    def accuracy(self, pred_label, labels):
        return accuracy_score(labels, pred_label)


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def encode_labels(labels):
    labels_reshaped = labels.reshape(len(labels), 1)
    encoder = OneHotEncoder(sparse=False)  # , categories='auto')
    return encoder.fit_transform(labels_reshaped)


mnist = np.load('datasets/mnist.pkl.npy')
train = mnist[0, 0]
train_size = 50
train_norm = normalize(train, axis=0)
train_sample = train_norm[:train_size]
train_labels = mnist[0, 1]
train_labels_sample = encode_labels(train_labels[:train_size])
validation = mnist[1, 0]
validation_labels = mnist[1, 1]
test = mnist[2, 0]
test_labels = mnist[2, 1]

batch_size = 5
epochs = 5
learning_rate = 0.0001
mlp = NN(hidden_dims=(450, 900))
mlp.verif_param_nn(mlp.total_param_nn)
weights, bias = mlp.initialize_weights(method='Zero')
# weights, bias = mlp.initialize_weights(method='Glorot')
# weights, bias = mlp.initialize_weights(method='Normal')
weights, bias = mlp.train(train_sample, train_labels_sample, weights, bias, epochs, learning_rate, batch_size)
_, _, predicted, chosen_class = mlp.forward(train_sample, weights, bias)
acc = mlp.accuracy(chosen_class, train_labels_sample)
print('Model accuracy: ', acc)

# epochs = 10
# learning_rate = 0.01
# mean_error = []
# for epoch in range(epochs):
#     cum_error = 0
#     for idx in range(len(train_sample)):
#         data = train_sample[idx]
#         labels = train_labels_sample[idx]
#
#         predicted = mlp.forward(data, weights, bias)
#         error = mlp.loss(predicted, labels)
#
#         # backprop missing
#         # weights, bias = mlp.update(grads_w, grads_b, weights, bias)
#
#         cum_error += error
#         # print("predicted = ", predicted)
#         # print("loss ", idx_loss)
#     mean_error.append(cum_error / len(train_sample))
#     print('epoch:', epoch, 'mean_error:', mean_error[epoch])
# plt.plot(mean_error)
# plt.savefig('mean_error.png')

# Test on the 2nd hidden layer by homework's instructions
# predictions = mlp.forward(data, weights, bias)
# mlp.test(weights[1], predictions[0], labels[0])
