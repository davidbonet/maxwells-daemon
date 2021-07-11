"""First real experiment - how well do we do on MNIST?"""

# import numpy as onp
from numpy.linalg import norm
import pickle
from collections import defaultdict

import jax.numpy as np
from jax import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import entropic_descent_deterministic, entropic_descent2, sgd_entropic
from maxwell_d.nn_utils import make_toy_cnn_funs, make_nn_funs
from maxwell_d.data import load_mnist, load_cifar10_2_classes

# ------ Problem parameters -------
layer_sizes = [784, 300, 10]
batch_size = 200
N_train = 10**3
N_tests = 10**3
num_channels = 5
# ------ Variational parameters -------
seed = 0
N_iter = 1000
alpha = 0.0001
init_scale = 0.1
# ------ Plot parameters -------
N_samples = 1
N_checkpoints = 50
thin = np.ceil(N_iter/N_checkpoints)

def neg_log_prior(w):
    return 0.5 * np.dot(w, w) / init_scale**2

def run():
    (train_images, train_labels),\
    (tests_images, tests_labels),\
    num_classes, IMAGE_SHAPE = load_cifar10_2_classes('plane','ship')
    # N_train = train_images.shape[0]
    # N_tests = tests_images.shape[0]
    # num_classes, IMAGE_SHAPE = load_mnist()
    (train_images, train_labels) = (train_images[:N_train], train_labels[:N_train])
    (tests_images, tests_labels) = (tests_images[:N_tests], tests_labels[:N_tests])
    parser, pred_fun, nllfun, frac_err = make_toy_cnn_funs(num_classes, num_channels, IMAGE_SHAPE, batch_size, seed)
    # parser, pred_fun, nllfun, frac_err = make_nn_funs(layer_sizes)
    N_param = len(parser.vect)
    print("Running experiment...")
    results = defaultdict(list)
    for i in range(N_samples):
        params = parser.vect

        def indexed_loss_fun(w, i_iter):
            rs = RandomState((seed, i, i_iter))
            idxs = rs.randint(N_train, size=batch_size)
            nll = nllfun(w, train_images[idxs], train_labels[idxs]) * N_train
            nlp = neg_log_prior(w)
            return nll + nlp
        gradfun = grad(indexed_loss_fun)

        def callback(x, t, entropy):
            results[("entropy", i)].append(entropy / N_train)
            # results[("v_norm", i)].append(norm(v) / np.sqrt(N_param))
            results[("minibatch_likelihood", i)].append(-indexed_loss_fun(x, t))
            results[("log_prior_per_dpt", i)].append(-neg_log_prior(x) / N_train)
            if t % thin != 0 and t != N_iter and t != 0: return
            results[('iterations', i)].append(t)
            results[("train_likelihood", i)].append(-nllfun(x, train_images, train_labels))
            results[("tests_likelihood", i)].append(-nllfun(x, tests_images, tests_labels))
            results[("tests_error", i)].append(frac_err(x, tests_images, tests_labels))
            print("Iteration {0:5} Train likelihood {1:2.4f}  Test likelihood {2:2.4f}" \
                  "  Test Err {3:2.4f}".format(t, results[("train_likelihood", i)][-1],
                                                  results[("tests_likelihood", i)][-1],
                                                  results[("tests_error",      i)][-1]))
        rs = RandomState((seed, i))
        sgd_entropic(gradfun, x_scale=params, N_iter=N_iter,
                     learn_rate=alpha, rs=rs, callback=callback, approx=True)
    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot_traces_and_mean(results, trace_type, X=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    if X is None:
        X = np.arange(len(results[(trace_type, 0)]))
    for i in range(N_samples):
        plt.plot(X, results[(trace_type, i)])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(trace_type)
    ax = fig.add_subplot(212)
    all_Y = [np.array(results[(trace_type, i)]) for i in range(N_samples)]
    plt.plot(X, sum(all_Y) / float(len(all_Y)))
    plt.savefig(trace_type + '.png')

def plot():
    print("Plotting results...")
    with open('results.pkl') as f:
          results = pickle.load(f)
    import matplotlib.pyplot as plt

    iters = results[('iterations', 0)]
    for i in range(N_samples):
        results[('marginal_likelihood', i)] = estimate_marginal_likelihood(
            results[("train_likelihood", i)],
            np.array(results[("entropy", i)])[iters])

    plot_traces_and_mean(results, 'entropy')
    # plot_traces_and_mean(results, 'v_norm')
    plot_traces_and_mean(results, 'minibatch_likelihood')
    plot_traces_and_mean(results, 'log_prior_per_dpt')
    plot_traces_and_mean(results, 'tests_likelihood', X=iters)
    plot_traces_and_mean(results, 'train_likelihood', X=iters)
    plot_traces_and_mean(results, 'tests_error',      X=iters)
    plot_traces_and_mean(results, 'marginal_likelihood', X=iters)

if __name__ == '__main__':
    results = run()
    f = open('results.pkl', 'wb')
    pickle.dump(results, f)
    plot()
