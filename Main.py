import numpy as np
from MLP import NN
from Utils import encode_labels, normalize, onehot
from random import shuffle

''' Import data '''
mnist = np.load('datasets/mnist.pkl.npy')
train = mnist[0, 0]
train_norm = normalize(train, axis=0)

indices = list(range(len(train)))
shuffle(indices)
train_size = indices[:20000]  # train.shape[0]

train_sample = train_norm[train_size]
train_labels = mnist[0, 1]
train_labels_sample = onehot(train_labels[train_size])

validation = mnist[1, 0]
validation_labels = mnist[1, 1]

test = mnist[2, 0]
test_labels = mnist[2, 1]

''' Hyper-parameters '''
batch_size = 64
epochs = 10
learning_rate = 10 ** (-4)

''' Model '''
mlp = NN(hidden_dims=(450, 450), method='Normal', verif_param=True)
average_loss = mlp.train(train_sample, train_labels_sample, epochs, learning_rate, batch_size)
print('Average loss: ', average_loss)
# _, _, _, _, predicted, chosen_class = mlp.forward(train_sample)
# acc = mlp.accuracy(chosen_class, train_labels_sample)
# print('Model accuracy: ', acc)

# mlp.test(train_labels_sample[-1], grad_test_h2)
