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
    def initialize_weights(self, n_hidden, dims):
        print('initialize_weights')
        weights = {}
        bias = {}
        for layer in range(len(dims) - 1):
            weights[layer] = np.random.rand(dims[layer], dims[layer + 1])
            bias[layer] = np.random.rand(dims[layer + 1])
        return weights, bias

    # %%
    def forward(self, input, labels, weights, bias):
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
        #print('activation')
        return input

    # %%
    def loss(self, prediction):
        print('loss')

    # %%
    def softmax(self, input):
        print('softmax')
        return input

    # %%
    def backward(self, cache, labels):
        print('backward')
        #SGD

    # %%
    def update(self, grads):
        print('update')

    # %%
    def train(self):
        #mettre la fct de cout cross_entrpy
        print('train')

    # %%
    def test(self):
        print('test')


mlp = NN()
mlp.verif_param_nn(mlp.total_param_nn)
weights, bias = mlp.initialize_weights(mlp.n_hidden, mlp.dims)

input = np.random.rand(1000, 784)
labels = np.random.randint(2, size=1000)
predicted = mlp.forward(input, labels, weights, bias)
print(predicted)

