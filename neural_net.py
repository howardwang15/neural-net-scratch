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


def one_hot_encoding(label):
    encoded = []
    for i in range(10):
        if label == i:
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded


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


def cross_entropy_loss(pred, actual):
    return -np.mean(actual * np.log(pred)/np.log(10))


def mean_squared_error(pred, actual):
    return np.mean((pred - actual)**2)


class Model:
    def __init__(self, inputs):
        self.w1 = create_weights(n_pixels, hidden_layer1_neurons)
        self.b1 = create_bias(hidden_layer1_neurons)

        self.w2 = create_weights(hidden_layer1_neurons, hidden_layer2_neurons)
        self.b2 = create_bias(hidden_layer2_neurons)

        self.w3 = create_weights(hidden_layer2_neurons, n_classes)
        self.b3 = create_bias(n_classes)

        self.h1 = np.dot(inputs, self.w1) + self.b1
        self.h1 = sigmoid(self.h1)

        self.h2 = np.dot(self.h1, self.w2) + self.b2
        self.h2 = sigmoid(self.h2)

        self.output = np.dot(self.h2, self.w3) + self.b3


images = np.asarray(images)
model = Model(images)


for i in range(n_epochs):
    train_batch = []
    batch_labels = []
    for j in range(batch_size):
        random_num = random.randrange(0, images.shape[0])
        train_batch.append(images[random_num])
        batch_labels.append(labels[random_num])

    train_batch = np.asarray(train_batch)
    batch_labels = np.asarray(batch_labels)
    for j in range(len(train_batch)):
        one_hot_label = np.asarray(one_hot_encoding(batch_labels[j]))
        error = cross_entropy_loss(model.output, one_hot_label)
        print(error)
        if j % batch_size == 0:
            print("hi")
