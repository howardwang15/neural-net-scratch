import pickle, gzip, numpy

def load_data():
    path = 'mnist.pkl.gz'
    f = gzip.open(path, 'rb')
    training, validation, testing = pickle.load(f, encoding='latin1')
    f.close()
    return training, validation, testing