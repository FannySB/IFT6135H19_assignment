import numpy as np
from MLP import NN
from Utils import onehot
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import random

''' Import data '''
mnist = np.load('datasets/mnist.pkl.npy')
# train = mnist[0, 0]
# # Shuffle the indices
# indices = list(range(train.shape[0]))
# shuffle(indices)
# shuffle_ind = indices[:10000]  # train.shape[0]
# # Train sample
# train_sample = train[shuffle_ind]
# train_labels = mnist[0, 1]
# train_labels_sample = train_labels[shuffle_ind]
# train_labels_sample_encoded = onehot(train_labels_sample)
# # Plot the distribution
# plt.hist(train_labels_sample)
# plt.savefig('labels_distribution.png')
# plt.clf()

# Validation set
# validation = mnist[1, 0]
# indices = list(range(validation.shape[0]))
# shuffle(indices)
# shuffle_ind = indices[:validation.shape[0]]  # validation.shape[0]
# validation_shuffled = validation[shuffle_ind]
# validation_labels = mnist[1, 1]
# validation_labels_shuffled = validation_labels[shuffle_ind]
# valid_lab_encoded = onehot(validation_labels_shuffled)
# # Test set
# test = mnist[2, 0]
# test_labels = mnist[2, 1]

''' Hyper-parameters '''
# batch_size = 64
# epochs = 5
# learning_rate = 10 ** (-1)

''' Model '''
# mlp = NN(hidden_dims=(512, 512), method='Glorot', verif_param=True)
# accuracy, average_loss = mlp.train(train_sample, train_labels_sample_encoded, epochs, learning_rate, batch_size)
# print('Training accuracy:', accuracy)
# _, valid_accuracy = mlp.validation(validation_shuffled, valid_lab_encoded)
# print('Validation accuracy:', valid_accuracy)
# print('Average loss: ', average_loss)
# _, _, _, _, predicted, chosen_class = mlp.forward(train_sample)
# # Confusion matrix
# cnf_matrix = mlp.confusion_matrix(train_labels_sample_encoded, predicted)
# df_cm = pd.DataFrame(cnf_matrix, range(10), range(10))
# sn.set(font_scale=1.4)  # for label size
# cm = sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 8})  # font size
# figure_cm = cm.get_figure()
# figure_cm.savefig('Confusion_matrix.png')

# class_names = np.array(list(range(10)))
# mlp.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                           title='Normalized confusion matrix')
# plt.savefig('Confusion_matrix.png')
# acc = mlp.accuracy(chosen_class, train_labels_sample)
# print('Model accuracy: ', acc)

# mlp.test(train_labels_sample[-1], grad_test_h2)

''' Question 1 '''
# batch_size = 64
# epochs = 10
# learning_rate = 10 ** (-1)
#
# mlp = NN(hidden_dims=(512, 512), method='Glorot', verif_param=True)
# accuracy, average_loss_glorot = mlp.train(train_sample, train_labels_sample_encoded, epochs, learning_rate, batch_size)
# mlp = NN(hidden_dims=(512, 512), method='Normal', verif_param=True)
# accuracy, average_loss_normal = mlp.train(train_sample, train_labels_sample_encoded, epochs, learning_rate, batch_size)
# mlp = NN(hidden_dims=(512, 512), method='Zero', verif_param=True)
# accuracy, average_loss_zeros = mlp.train(train_sample, train_labels_sample_encoded, epochs, learning_rate, batch_size)
#
# glorot = plt.plot(average_loss_glorot, label='Glorot')
# normal = plt.plot(average_loss_normal, label='Normal')
# zero = plt.plot(average_loss_zeros, label='Zero')
# plt.legend()
# plt.savefig('Average_loss.png')
# plt.clf()
#
# glorot2 = plt.plot(average_loss_glorot, label='Glorot')
# normal2 = plt.plot(average_loss_normal, label='Normal')
# plt.legend()
# plt.savefig('Average_loss_Glorot_Normal.png')
# plt.clf()

''' Question 2 '''
# Hyper-parameters to search on
# h1 = [415, 500, 585, 675]
# h2 = [415, 500, 585, 675]
# bs = [16, 32, 64, 128]
# lr = [10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4)]
# epoch = [2, 3, 4, 5]
#
# # Random search
# liste = list(range(8))
# random_search = []
# for i in liste:
#     random_search.append(list(np.random.choice(4, 5)))
#
# acc = []
# for i in liste:
#     h1_random = h1[random_search[i][0]]
#     h2_random = h2[random_search[i][1]]
#     bs_random = bs[random_search[i][2]]
#     lr_random = lr[random_search[i][3]]
#     epoch_random = epoch[random_search[i][4]]
#     print('Random search :', i)
#     print(f'h1: {h1_random}, h2: {h2_random}, bs: {bs_random}, lr: {lr_random}, epoch: {epoch_random}')
#     mlp = NN(hidden_dims=(h1_random, h2_random), method='Glorot', verif_param=True)
#     _, train_acc = mlp.train(train_sample, train_labels_sample_encoded,
#                              epoch_random, lr_random, bs_random)
#     print(f'Train accuracy: {train_acc}')
#     _, valid_acc = mlp.validation(validation_shuffled, valid_lab_encoded)
#     print(f'Validation accuracy: {valid_acc}\n')
#     acc.append(valid_acc)

''' Question 3 '''
# Train sample
train = mnist[0, 0]
train_sample = train[2]
train_labels = mnist[0, 1]
train_labels_sample = train_labels[2]
train_labels_sample_encoded = onehot(train_labels_sample)

np.random.seed(123)
mlp = NN(hidden_dims=(10, 10), method='Glorot', verif_param=False, axis=0)
mlp.finite_difference(train_sample, train_labels_sample_encoded)
