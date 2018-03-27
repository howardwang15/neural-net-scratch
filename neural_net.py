import numpy as np
import random
from mnist import MNIST

#constants for each image
n_pixels = 784
n_classes = 10

#hyperparameters
hidden_layer1_neurons = 500
hidden_layer2_neurons = 500
learning_rate = 0.01
batch_size = 100
n_epochs = 5000


#read mnist data
mndata = MNIST('C:\\Users\\joy_l\\PycharmProjects\\MNIST_data')
images, labels = mndata.load_training()
index = random.randrange(0, len(images) - 5000)

pic = images[index]


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def create_weights(input_size, output_size):
    a = np.empty([input_size, output_size])
    for i in range(0, output_size):
        weight = random.uniform(-0.5, 0.5)
        np.append(a, weight)
    return a


def create_bias(output_size):
    a = np.full((1, output_size), 0.1)
    return a


def mean_squared_error(pred, actual):
    return (pred - actual)**2


def create_model(inputs):
    w1 = create_weights(n_pixels, hidden_layer1_neurons)
    b1 = create_bias(hidden_layer1_neurons)

    w2 = create_weights(hidden_layer1_neurons, hidden_layer2_neurons)
    b2 = create_bias(hidden_layer2_neurons)

   # w3 =
    h1 = np.dot(inputs, w1) + b1




for i in range(n_epochs):
    for i in range(0, len(images)):
        if i % batch_size == 0:
            print("hl")


#print(b)
