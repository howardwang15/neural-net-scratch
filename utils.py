import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def softmax(prediction):
    return np.exp(prediction)/np.sum(np.exp(prediction))

def one_hot_encoding(x):
    hot = np.eye(10)[np.array(x).reshape(-1)]
    return hot.reshape(list(x.shape)+[10])
