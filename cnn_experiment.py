"""First real experiment - how well do we do on MNIST?"""
import matplotlib.pyplot as plt
from matplotlib import rc
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
# layer_sizes = [784, 300, 10]
batch_size = 50
N_train = 10000
N_tests = 2000
num_channels = 5
# ------ Variational parameters -------
seed = 3
init_scale = 0.1
N_iter = 200
alpha = 0.00001
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
    parser, pred_fun, nllfun, frac_err, nnk_loo_jax = make_toy_cnn_funs(num_classes, num_channels, IMAGE_SHAPE, batch_size, seed)
    # parser, pred_fun, nllfun, frac_err = make_nn_funs(layer_sizes)
    # alpha = alpha / N_train
    N_param = len(parser.vect)
    print("Running experiment...", flush=True)
    params = parser.vect

    def indexed_loss_fun(w, i_iter):
        rs = RandomState((seed, i, i_iter))
        idxs = rs.randint(N_train, size=batch_size)
        nll = nllfun(w, train_images[idxs], train_labels[idxs]) * N_train
        nlp = neg_log_prior(w)
        return nll + nlp
    gradfun = grad(indexed_loss_fun)

    def callback(x, t, entropy):
        results["entropy_per_dpt"     ].append(entropy / N_train)
        results["minibatch_likelihood"].append(-indexed_loss_fun(x, t))
        results["log_prior_per_dpt"   ].append(-neg_log_prior(x) / N_train)
        if t % thin != 0 and t != N_iter and t != 0: return
        results["iterations"      ].append(t)
        results["train_likelihood"].append(-nllfun(x, train_images, train_labels))
        results["tests_likelihood"].append(-nllfun(x, tests_images, tests_labels))
        results["train_error"     ].append(frac_err(x, train_images, train_labels))
        results["tests_error"     ].append(frac_err(x, tests_images, tests_labels))
        results["marg_likelihood" ].append(estimate_marginal_likelihood(
            results["train_likelihood"][-1], results["entropy_per_dpt"][-1]))
        results["nnk_loo_layer"   ].append(nnk_loo_jax(x, train_images, train_labels))
        
        print("Iteration {0:5} Train lik {1:2.4f}  Test lik {2:2.4f}" \
              "  Marg lik {3:2.4f}  Test Err {4:2.4f}   Entropy {5:2.4f}".format(
                  t, results["train_likelihood"][-1],
                  results["tests_likelihood"][-1],
                  results["marg_likelihood" ][-1],
                  results["tests_error"     ][-1],
                  results["entropy_per_dpt" ][-1]), flush=True)
    
    all_results = []
    for i in range(N_samples):
        results = defaultdict(list)
        rs = RandomState((seed, i))
        # sgd_entropic(gradfun, x_scale=params, N_iter=N_iter,
        #                 learn_rate=alpha, rs=rs, callback=callback, approx=True)
        sgd_entropic(gradfun, x_scale=np.full(N_param, init_scale), N_iter=N_iter,
                        learn_rate=alpha, rs=rs, callback=callback, approx=True)
        all_results.append(results)
    
    return all_results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print("Plotting results...", flush=True)
    with open('results.pkl', 'rb') as f:
          results = pickle.load(f)

    first_results = results[0]
    # Diagnostic plots of everything for us.
    for key in first_results:
        plot_traces_and_mean(results, key)

    # Nice plots for paper.
    rc('font',**{'family':'serif'})
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    plt.plot(first_results["iterations"], first_results["train_error"], 'b', label="Train error")
    plt.plot(first_results["iterations"], first_results["tests_error"], 'g', label="Test error")
    best_marg_like = first_results["iterations"][np.argmax(np.array(first_results["marg_likelihood"]))]
    plt.axvline(x=best_marg_like, color='black', ls='dashed', zorder=2)
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    ax.set_ylabel('Error')

    ax = fig.add_subplot(212)
    plt.plot(first_results["iterations"], first_results["marg_likelihood"], 'r', label="Marginal likelihood")
    plt.axvline(x=best_marg_like, color='black', ls='dashed', zorder=2)
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    ax.set_ylabel('Marginal likelihood')
    ax.set_xlabel('Training iteration')
    #low, high = ax.get_ylim()
    #ax.set_ylim([0, high])

    fig.set_size_inches((5,3.5))
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    plt.savefig('marglik.pdf', pad_inches=0.05, bbox_inches='tight')
    
def plot_traces_and_mean(results, trace_type, X=None):
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    if X is None:
        X = np.arange(len(results[0][trace_type]))
    for i in range(N_samples):
        plt.plot(X, results[i][trace_type])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(trace_type)
    ax = fig.add_subplot(212)
    all_Y = [np.array(results[i][trace_type]) for i in range(N_samples)]
    plt.plot(X, sum(all_Y) / float(len(all_Y)))
    plt.savefig(trace_type + '.png')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    plot()
