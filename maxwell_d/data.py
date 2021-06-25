import os
import gzip
import struct
import array
import numpy as np
import numpy.random as npr
import pickle
from tensorflow import keras

def datapath(fname):
    datadir = os.path.expanduser('data')
    return os.path.join(datadir, fname)

def mnist():

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)
    train_images = parse_images(datapath('mnist/train-images-idx3-ubyte.gz'))
    train_labels = parse_labels(datapath('mnist/train-labels-idx1-ubyte.gz'))
    test_images  = parse_images(datapath('mnist/t10k-images-idx3-ubyte.gz'))
    test_labels  = parse_labels(datapath('mnist/t10k-labels-idx1-ubyte.gz'))

    return train_images, train_labels, test_images, test_labels

def lecun_gz_to_pickle():
    data = mnist()
    with open(datapath("mnist/mnist_data.pkl"), "w") as f:
        pickle.dump(data, f, 1)

def load_data(normalize=False):
    with open(datapath("mnist/mnist_data.pkl"), 'rb') as f:
        train_images, train_labels, test_images, test_labels = pickle.load(f, encoding='latin1')


    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    if normalize:
        train_mean = np.mean(train_images, axis=0)
        train_images = train_images - train_mean
        test_images = test_images - train_mean
    return train_images, train_labels, test_images, test_labels, N_data

def load_data_subset(*args):
    train_images, train_labels, test_images, test_labels, _ = load_data(normalize=True)
    all_images = np.concatenate((train_images, test_images), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)
    datapairs = []
    start = 0
    for N in args:
        end = start + N
        datapairs.append((all_images[start:end], all_labels[start:end]))
        start = end
    return datapairs

def load_data_dicts(*args):
    datapairs = load_data_subset(*args)
    return [{"X" : dat[0], "T" : dat[1]} for dat in datapairs]


def load_boston_housing(train_frac=0.5, rs=npr.RandomState(0)):
    data = np.loadtxt(datapath('boston_housing.txt'))
    X = data[:,:-1]
    y = data[:,-1][:, None]

    # Create train and test sets with 90% and 10% of the data
    permutation = rs.choice(range(X.shape[0]), X.shape[0], replace=False)
    size_train = np.round(X.shape[0] * train_frac)
    train_ixs = permutation[0:size_train]
    test_ixs = permutation[size_train:]

    X = X - np.mean(X[train_ixs, :], axis=0, keepdims=True)
    X = X / np.std(X[train_ixs, :], axis=0, keepdims=True)

    y_mean = np.mean(y[train_ixs])
    y = y - y_mean
    y_std = np.std(y[train_ixs])
    y = y / y_std
    print("Scaling boston housing prices by {0}".format(y_std))

    def unscale_y(y):
        return y * y_std

    return X[train_ixs,:], y[train_ixs], X[test_ixs,:], y[test_ixs], unscale_y

def load_mnist():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    IMAGE_SHAPE = [28, 28, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), num_classes, IMAGE_SHAPE

def load_cifar10():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    IMAGE_SHAPE = [32, 32, 3]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), num_classes, IMAGE_SHAPE

def load_cifar10_2_classes(c1, c2):
    class_names = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = 2
    c1 = class_names.index(c1)
    c2 = class_names.index(c2)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    IMAGE_SHAPE = [32, 32, 3]
    train_filter = np.where((y_train == c1 ) | (y_train == c2))[0]
    test_filter = np.where((y_test == c1) | (y_test == c2))[0]
    y_train = np.where(y_train == c1, 0, y_train)
    y_test = np.where(y_test == c1, 0, y_test)
    y_train = np.where(y_train == c2, 1, y_train)
    y_test = np.where(y_test == c2, 1, y_test)
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), num_classes, IMAGE_SHAPE


if __name__=="__main__":
    lecun_gz_to_pickle()
