import numpy as np
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict

from jax import random
from jax.experimental import stax
from jax.experimental.stax import (Conv, Dense, Flatten, Relu, LogSoftmax, Softmax, MaxPool)
from jax.nn.initializers import he_uniform

class WeightsParser(object):
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def set(self, vect, name, val):
        idxs, shape = self.idxs_and_shapes[name]
        if isinstance(val, np.ndarray):
            vect[idxs] = val.ravel()
        else:
            vect[idxs] = val  # Can't unravel a float.

class VectorParser(object):
    def __init__(self):
        self.idxs_and_shapes = OrderedDict()
        self.vect = np.zeros((0,))

    def add_shape(self, name, shape):
        start = len(self.vect)
        size = np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, start + size), shape)
        self.vect = np.concatenate((self.vect, np.zeros(size)), axis=0)
        
    def add_shape_and_values(self, name, array):
        start = len(self.vect)
        size = np.prod(array.shape)
        self.idxs_and_shapes[name] = (slice(start, start + size), array.shape)
        self.vect = np.concatenate((self.vect, array.reshape(size)), axis=0)

    def new_vect(self, vect):
        assert vect.size == self.vect.size
        new_parser = self.empty_copy()
        new_parser.vect = vect
        return new_parser

    def empty_copy(self):
        """Creates a parser with a blank vector."""
        new_parser = VectorParser()
        new_parser.idxs_and_shapes = self.idxs_and_shapes.copy()
        new_parser.vect = None
        return new_parser

    def as_dict(self):
        return {k : self[k] for k in self.names}

    @property
    def names(self):
        return self.idxs_and_shapes.keys()

    def __getitem__(self, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(self.vect[idxs], shape)

    def __setitem__(self, name, val):
        if isinstance(val, list): val = np.array(val)
        if name not in self.idxs_and_shapes:
            self.add_shape(name, val.shape)

        idxs, shape = self.idxs_and_shapes[name]
        self.vect[idxs].reshape(shape)[:] = val

def fill_parser(parser, items):
    """Build a vector by assigning each block the corresponding value in
       the items vector."""
    partial_vects = [np.full(parser[name].size, items[i])
                     for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

class BatchList(list):
    def __init__(self, N_total, N_batch):
        start = 0
        while start < N_total:
            self.append(slice(start, start + N_batch))
            start += N_batch
        self.all_idxs = slice(0, N_total)

def logsumexp(X, axis):
    max_X = jnp.max(X)
    return max_X + jnp.log(jnp.sum(jnp.exp(X - max_X), axis=axis, keepdims=True))

def logit(x): return 1 / (1 + jnp.exp(-x))
def inv_logit(y): return -jnp.log( 1/y - 1)
def d_logit(x): return logit(x) * (1 - logit(x))

def make_nn_funs(layer_sizes):
    parser = VectorParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_shape(('weights', i), shape)
        parser.add_shape(('biases', i), (1, shape[1]))

    def predictions(W_vect, X):
        """Outputs normalized log-probabilities."""
        W = parser.new_vect(W_vect)
        cur_units = X
        N_layers = len(layer_sizes) - 1
        for i in range(N_layers):
            cur_W = W[('weights', i)]
            cur_B = W[('biases',  i)]
            cur_units = jnp.dot(cur_units, cur_W) + cur_B
            if i == (N_layers - 1):
                cur_units = cur_units - logsumexp(cur_units, axis=1)
            else:
                cur_units = jnp.tanh(cur_units)
        return cur_units

    def loss(W_vect, X, T):
        # log_prior = - 0.5 * L2_reg * np.dot(W_vect, W_vect)
        return - np.sum(predictions(W_vect, X) * T) / X.shape[0]

    def frac_err(W_vect, X, T):
        preds = np.argmax(predictions(W_vect, X), axis=1)
        return np.mean(np.argmax(T, axis=1) != preds)

    return parser, predictions, loss, frac_err

def make_regression_nn_funs(layer_sizes):
    """Same as above but outputs mean predictions.
    Loss is normalized Gaussian pdfs with variance 1."""
    parser = VectorParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_shape(('weights', i), shape)
        parser.add_shape(('biases', i), (1, shape[1]))

    def predictions(W_vect, X):
        """Outputs mean prediction."""
        W = parser.new_vect(W_vect)
        cur_units = X
        N_layers = len(layer_sizes) - 1
        for i in range(N_layers):
            cur_W = W[('weights', i)]
            cur_B = W[('biases',  i)]
            cur_units = np.dot(cur_units, cur_W) + cur_B
            if i < (N_layers - 1):
                cur_units = np.tanh(cur_units)
        return cur_units

    def loss(W_vect, X, T):
        """Outputs average normalized log-probabilities of Gaussians with variance of 1"""
        # log_prior = - 0.5 * L2_reg * np.dot(W_vect, W_vect)
        return np.mean((predictions(W_vect, X) - T)**2) + 0.5*np.log(2*np.pi)

    def rmse(W_vect, X, T):
        return np.sqrt(np.mean((predictions(W_vect, X) - T)**2))

    return parser, predictions, loss, rmse

def nice_layer_name(weight_key):
    """Takes a tuple like ('weights', 2) and returns a nice string like "2nd layer weights"
       for use in plots and legends."""
    return "Layer {num} {name}".format(num=weight_key[1] + 1, name=weight_key[0])

def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28,28),
                cmap=matplotlib.cm.binary, vmin=None):

    """iamges should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.ceil(float(N_images) / ims_per_row)
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                            (digit_dimensions[0] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i / ims_per_row  # Integer division.
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0])*row_ix
        col_start = padding + (padding + digit_dimensions[0])*col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[0]] \
            = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax

def make_toy_cnn_funs(num_classes, num_channels, image_shape, batch_size, seed=0):
    key = random.PRNGKey(seed)
    init_fun, conv_net = stax.serial(# Layer 0
                                    Conv(num_channels, (5, 5), strides=None, padding="VALID", W_init=he_uniform(), b_init=he_uniform()),
                                    Relu,
                                    # Layer 1
                                    Conv(num_channels, (5, 5), strides=None, padding="VALID", W_init=he_uniform(), b_init=he_uniform()),
                                    Relu,
                                    MaxPool((2, 2), strides=(2, 2), padding="SAME"),
                                    # Layer 2
                                    Conv(num_channels, (5, 5), strides=None, padding="VALID", W_init=he_uniform(), b_init=he_uniform()),
                                    Relu,
                                    # Layer 3
                                    Conv(num_channels, (3, 3), strides=None, padding="VALID", W_init=he_uniform(), b_init=he_uniform()),
                                    Relu,
                                    MaxPool((2, 2), strides=(2, 2), padding="SAME"),
                                    # Output
                                    Flatten,
                                    Dense(num_classes),
                                    LogSoftmax)
    _, params = init_fun(key, (batch_size, image_shape[0], image_shape[1], image_shape[2]))
    trainable_layers = [0,2,5,7,11]
    num_layers = 13
    parser = VectorParser()
    for i, layer in enumerate(trainable_layers):
        weights, biases = params[layer]
        # parser.add_shape(('weights', i), weights.shape)
        # parser.add_shape(('biases', i), biases.shape)
        parser.add_shape_and_values(('weights', i), weights)
        parser.add_shape_and_values(('biases', i), biases)

    def predictions(W_vect, images):
        """Outputs normalized log-probabilities."""
        W = parser.new_vect(W_vect)
        cur_params = []
        it_trainable = 0
        for i in range(num_layers):
            if i in trainable_layers:
                weights = W[('weights', it_trainable)]
                biases = W[('biases', it_trainable)]
                cur_params.append((weights, biases))
                it_trainable += 1
            else:
                cur_params.append(())
        return conv_net(cur_params, images)
    
    def loss(W_vect, X, T):
        return - np.sum(predictions(W_vect, X) * T) / X.shape[0]

    def frac_err(W_vect, X, T):
        preds = np.argmax(predictions(W_vect, X), axis=1)
        return np.mean(np.argmax(T, axis=1) != preds)

    return parser, predictions, loss, frac_err