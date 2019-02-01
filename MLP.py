import numpy as np


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
        assert (
            not (total_param_nn < 1000000 & total_param_nn > 500000),
            'ERROR! Number of parameter in NN: ' + str(total_param_nn))

    # %%
    def initialize_weights(self):
        print('initialize_weights')
        weights = {}
        bias = {}
        for layer in range(len(self.dims) - 1):
            weights[layer] = np.random.rand(self.dims[layer], self.dims[layer + 1])
            bias[layer] = np.random.rand(self.dims[layer + 1])
        return weights, bias

    # %%
    def forward(self, input, weights, bias):
        print('forward')
        foward_pass = {}
        for layer in range(len(weights)):
            if layer == 0:
                for j in range(len(input)):
                    foward_pass[j] = self.activation(np.dot(weights[layer].transpose(), input[j]) + bias[layer])
                print(foward_pass[j].shape)
            else:
                for j in range(len(input)):
                    foward_pass[j] = self.activation(np.dot(weights[layer].transpose(), foward_pass[j]) + bias[layer])
                print(foward_pass[j].shape)
            print(len(foward_pass.keys()))
        return self.softmax(foward_pass)

    # %%
    def activation(self, input):
        return np.maximum(0, input)

    # %%
    def loss(self, prediction, labels):
        print('loss')
        pred = np.amax(prediction, axis=1)
        return -np.sum(labels * np.log(pred))

    # %%
    def softmax(self, input):
        print('softmax')
        input_array = np.array(list(input.values()))
        # input_array  -= np.max(input_array) #To reduce the values of the output. Not let it go to infinity.
        denominator = np.sum(np.exp(input_array))
        return input_array / denominator

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


mlp = NN()
mlp.verif_param_nn(mlp.total_param_nn)
weights, bias = mlp.initialize_weights()

input = np.random.rand(1000, 784)
labels = np.random.randint(2, size=1000)
predicted = mlp.forward(input, weights, bias)
ce = mlp.loss(predicted, labels)
print(ce)