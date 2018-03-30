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
images, labels = mndata.load_training() #load the training data

def one_hot_encoding(label): #convert a label (ie 0-9 digit into a one hot encoded vector)
    encoded = []
    for i in range(10):
        if label == i:
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded


def sigmoid(x): #sigmoid activation function
    return 1.0/(1 + np.exp(-x))


def softmax(x):
    b = []
    for i in x:
        a = []
        temp = i
        for j in i:
            a.append(np.exp(j)/float(np.sum(np.exp(i))))
        b.append(a)
    return np.asarray(b)


def create_weights(input_size, output_size):
    w = np.asarray(np.random.rand(input_size, output_size) - 0.5)
    return w


def create_bias(output_size):
    a = np.full((1, output_size), 0.1)
    return a


def cross_entropy_loss(pred, actual):
    pred = pred.tolist()
    losses = []
    for i in range(len(pred)):
        avg_loss = -np.mean(actual[i] * np.log(pred[i])/np.log(10))
        losses.append(avg_loss)
    losses = np.asarray(losses)

    return np.mean(losses)


class Model:
    def __init__(self):
        self.w1 = create_weights(n_pixels, hidden_layer1_neurons)
        self.b1 = create_bias(hidden_layer1_neurons)

        self.w2 = create_weights(hidden_layer1_neurons, hidden_layer2_neurons)
        self.b2 = create_bias(hidden_layer2_neurons)

        self.w3 = create_weights(hidden_layer2_neurons, n_classes)
        self.b3 = create_bias(n_classes)


    def predict(self, inputs):
        self.h1 = np.dot(inputs, self.w1) + self.b1
        self.h1 = sigmoid(self.h1)

        self.h2 = np.dot(self.h1, self.w2) + self.b2
        self.h2 = sigmoid(self.h2)

        self.output = np.dot(self.h2, self.w3) + self.b3
        self.output = softmax(self.output)

        return self.output



images = np.asarray(images)

train_batch = []  # select images to be trained
batch_labels = []  # the labels of those images
for j in range(batch_size):
    random_num = random.randrange(0, images.shape[0])
    train_batch.append(images[random_num])
    batch_labels.append(one_hot_encoding(labels[random_num]))

prediction = Model()

for i in range(n_epochs): #train for specified num of epochs
    output = prediction.predict(train_batch)

    loss = cross_entropy_loss(output, batch_labels)
    assert(len(train_batch) == len(batch_labels))


