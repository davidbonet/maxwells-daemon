import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.experimental.stax import (Conv, Dense, Flatten, Relu, LogSoftmax, MaxPool)
from jax.nn.initializers import he_uniform

def make_cnn_funs(num_classes, num_channels, batch_size, seed=0):
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
    _, params = init_fun(key, (batch_size, 32, 32, 3))
    
    def loss(params, images, targets):
        preds = conv_net(params, images)
        return - np.sum(preds * targets) / images.shape[0]
        
    return conv_net, params, loss