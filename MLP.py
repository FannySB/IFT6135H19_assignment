import numpy as np
from math import sqrt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import itertools


class NN(object):

    def __init__(self, hidden_dims=(350, 700), n_hidden=2, method='Glorot', verif_param=True):
        self.W1 = []
        self.W2 = []
        self.W3 = []
        self.b1 = []
        self.b2 = []
        self.b3 = []
        self.parameters = []
        self.n_hidden = n_hidden
        h1 = hidden_dims[0]
        h2 = hidden_dims[1]
        self.dim_out = 10
        features = 784
        self.dims = [features, h1, h2, self.dim_out]
        self.total_param_nn = features * h1 + h2 * h1 + self.dim_out * h1 + h1 + h2 + self.dim_out
        if verif_param:
            self.verif_param_nn(self.total_param_nn)
        self.initialize_weights(method)

    def verif_param_nn(self, total_param_nn):
        print('Number of parameter in NN: ', total_param_nn)
        if not (total_param_nn < 1000000 and total_param_nn > 500000):
            raise AssertionError('ERROR! Number of parameter in NN: ' + str(total_param_nn))

    def initialize_weights(self, method='Normal'):
        if method == 'Normal':
            self.W1 = np.random.randn(self.dims[1], self.dims[0]) * 10 ** (-1)  # * np.sqrt(2 / self.dims[1])
            self.W2 = np.random.randn(self.dims[2], self.dims[1]) * 10 ** (-1)  # * np.sqrt(2 / self.dims[2])
            self.W3 = np.random.randn(self.dims[3], self.dims[2]) * 10 ** (-1)  # * np.sqrt(2 / self.dims[3])
            self.b1 = np.zeros((self.dims[1],), dtype=float)
            self.b2 = np.zeros((self.dims[2],), dtype=float)
            self.b3 = np.zeros((self.dims[3],), dtype=float)
            self.parameters = [self.b1, self.W1, self.b2, self.W2, self.b3, self.W3]
        elif method == 'Zero':
            self.W1 = np.zeros((self.dims[1], self.dims[0]))
            self.W2 = np.zeros((self.dims[2], self.dims[1]))
            self.W3 = np.zeros((self.dims[3], self.dims[2]))
            self.b1 = np.zeros((self.dims[1],), dtype=float)
            self.b2 = np.zeros((self.dims[2],), dtype=float)
            self.b3 = np.zeros((self.dims[3],), dtype=float)
            self.parameters = [self.b1, self.W1, self.b2, self.W2, self.b3, self.W3]
        else:
            d_layer1 = sqrt(6 / (self.dims[1] + self.dims[0]))
            d_layer2 = sqrt(6 / (self.dims[2] + self.dims[1]))
            d_layer3 = sqrt(6 / (self.dims[3] + self.dims[2]))
            self.W1 = np.random.uniform(-d_layer1, d_layer1, (self.dims[1], self.dims[0]))
            self.W2 = np.random.uniform(-d_layer2, d_layer2, (self.dims[2], self.dims[1]))
            self.W3 = np.random.uniform(-d_layer3, d_layer3, (self.dims[3], self.dims[2]))
            self.b1 = np.zeros((self.dims[1],), dtype=float)
            self.b2 = np.zeros((self.dims[2],), dtype=float)
            self.b3 = np.zeros((self.dims[3],), dtype=float)
            self.parameters = [self.b1, self.W1, self.b2, self.W2, self.b3, self.W3]

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
        return (labels * (-np.log(prediction))).sum(axis=1).mean()

    def accuracy(self, pred_label, labels):
        labels_decoded = labels.argmax(axis=1)
        return (pred_label == labels_decoded).mean()

    def forward(self, input):
        h1_preact = np.dot(input, np.transpose(self.W1)) + self.b1
        h1 = self.activation(h1_preact)
        h2_preact = np.dot(h1, np.transpose(self.W2)) + self.b2
        h2 = self.activation(h2_preact)
        out_preact = np.dot(h2, np.transpose(self.W3)) + self.b3
        out = self.softmax(out_preact)
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
        dl_dh2 = np.dot(dl_dsoftmax, self.W3)
        dl_dh2_relu = (h2_preact > 0) * dl_dh2

        # Gradient of W^(2) and b^(2)
        grad_w2 = np.dot(dl_dh2_relu.T, h1) / batch_size
        grad_b2 = dl_dh2_relu.mean(axis=0)

        # Derivative for hidden 1
        dl_dh1 = np.dot(dl_dh2, self.W2)
        dl_dh1_relu = (h1_preact > 0) * dl_dh1

        # Gradient of W^(1) and b^(1)
        grad_w1 = np.dot(dl_dh1_relu.T, input) / batch_size
        grad_b1 = dl_dh1_relu.mean(axis=0)

        # Update the parameters
        grads = [grad_b1, grad_w1, grad_b2, grad_w2, grad_b3, grad_w3]
        self.update(grads, learning_rate)

    def update(self, grads, learning_rate):
        for parameter, gradient in zip(self.parameters, grads):
            parameter -= learning_rate * gradient

    def train(self, input, labels, epochs, learning_rate, batch_size):
        # print('Training....')
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
            # print(f'It took : {(end_epoch - start_epoch):.2f} seconds for epoch {epoch}')
            # print(f'Error for epoch {epoch}: {loss:.2f}')
            acc = self.accuracy(pred_label, labels)
            accuracy.append(acc)
            # print(f'Model accuracy: {acc}\n')

            # Confusion matrix
            cnf_matrix = self.confusion_matrix(labels, predicted)
            # print(cnf_matrix)

        # plt.plot(losses)
        # plt.savefig('losses.png')
        # plt.clf()
        # plt.plot(accuracy)
        # plt.savefig('accuracy.png')
        # plt.clf()
        end = time.time()
        print(f'It took : {(end - start)} seconds')
        return losses, accuracy

    def validation(self, input, labels):
        _, _, _, _, _, pred_label = self.forward(input)
        accuracy = self.accuracy(pred_label, labels)
        return pred_label, accuracy

    def confusion_matrix(self, y_true, y_pred):
        # Transform into array of classes
        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.clf()

    def finite_difference(self, labels, grad_test_h2):
        print('test')
        # Const chosen
        N_vector = [10, 500, 1000, 125000, 6250000]
        grads_all = []
        p = min(10, len(self.W2))
        target_weights = self.W2[:p]
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
