"""Generates plot showing properties of ensembles."""
import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import pickle
from collections import defaultdict
from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import sgd_entropic_damped
from maxwell_d.nn_utils import make_nn_funs
from maxwell_d.data import load_data_subset

# ------ Problem parameters -------

N_train = 50000
batch_size = 250
N_tests = 1000
# ------ Variational parameters -------
seed = 0

N_iter = 3000
alpha = 0.1 / N_train

widths = [0.1, 1, 10, 100]

# ------ Plot parameters -------
N_samples = 10
N_checkpoints = 10
thin = np.ceil(N_iter/N_checkpoints)


def run():
    all_results = []
    for width in widths:

        init_scale = np.min( 0.5 / np.sqrt(width), 0.5/np.sqrt(784))
        def neg_log_prior(w):
            D = len(w)
            return 0.5 * D * np.log(2*np.pi) + 0.5 * np.dot(w, w) / init_scale**2 + D * np.log(init_scale)

        (train_images, train_labels),\
        (tests_images, tests_labels) = load_data_subset(N_train, N_tests)
        layer_sizes = [784, 100, 10]
        parser, pred_fun, nllfun, frac_err = make_nn_funs(layer_sizes)
        N_param = len(parser.vect)
        print "Number of parameters in model: ", N_param

        def indexed_loss_fun(w, i_iter):
            rs = RandomState((seed, i_iter))
            idxs = rs.randint(N_train, size=batch_size)
            nll = nllfun(w, train_images[idxs], train_labels[idxs]) * N_train
            #nlp = neg_log_prior(w)
            return nll# + nlp
        gradfun = grad(indexed_loss_fun)

        def callback(x, t, entropy):
            results["entropy_per_dpt"     ].append(entropy / N_train)
            results["x_rms"               ].append(np.sqrt(np.mean(x * x)))
            results["minibatch_likelihood"].append(-indexed_loss_fun(x, t))
            results["log_prior_per_dpt"   ].append(-neg_log_prior(x) / N_train)
            if t % thin != 0 and t != N_iter and t != 0: return
            results["iterations"      ].append(t)
            results["train_likelihood"].append(-nllfun(x, train_images, train_labels))
            results["tests_likelihood"].append(-nllfun(x, tests_images, tests_labels))
            results["tests_error"     ].append(frac_err(x, tests_images, tests_labels))
            results["marg_likelihood" ].append(estimate_marginal_likelihood(
                results["log_prior_per_dpt"][-1],
                results["train_likelihood" ][-1],
                results["entropy_per_dpt"  ][-1]))

            print "Iteration {0:5} Train lik {1:2.4f}  Test lik {2:2.4f}" \
                  "  Marg lik {3:2.4f}  Test err {4:2.4f}".format(
                      t, results["train_likelihood"][-1],
                      results["tests_likelihood"][-1],
                      results["marg_likelihood" ][-1],
                      results["tests_error"     ][-1])

        results = defaultdict(list)
        rs = RandomState((seed))
        sgd_entropic_damped(gradfun, np.full(N_param, init_scale), N_iter,
                            alpha, rs, callback, width=width)
        all_results.append(results)

    return all_results

def estimate_marginal_likelihood(prior, likelihood, entropy):
    return prior + likelihood + entropy

def plot():
    print "Plotting results..."
    with open('results.pkl') as f:
          all_results = pickle.load(f)

    #for key in all_results[0]:
    #    plot_traces_and_mean(all_results, key)


    fig = plt.figure(0); fig.clf()
    X = all_results[0]["iterations"]
    ax = fig.add_subplot(211)
    final_train = [all_results[i]["train_likelihood"][-1] for i, w in enumerate(widths)]
    final_tests = [all_results[i]["tests_likelihood"][-1] for i, w in enumerate(widths)]
    plt.plot(final_train, label='training likelihood')
    plt.plot(final_tests, label='test likelihood')
    plt.xticks(range(len(widths)), widths)
    ax.legend(numpoints=1, loc=3, frameon=False, prop={'size':'12'})
    ax.set_ylabel('Test Likelihood')


    ax = fig.add_subplot(212)
    final_marg = [all_results[i]["marg_likelihood"][-1] for i, w in enumerate(widths)]
    plt.plot(final_marg, label='marginal likelihood')
    ax.set_ylabel('Marginal likelihood')
    ax.set_xlabel('Training iteration')
    plt.xticks(range(len(widths)), widths)
    #low, high = ax.get_ylim()
    #ax.set_ylim([0, high])
    # plt.show()
    fig.set_size_inches((5,3.5))
    #ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    plt.savefig('vary_widths.pdf', pad_inches=0.05, bbox_inches='tight')


    # # Nice plots for paper.
    # rc('font',**{'family':'serif'})
    # fig = plt.figure(0); fig.clf()
    # X = all_results[0]["iterations"]
    # ax = fig.add_subplot(212)
    # for i, w in enumerate(widths):
    #     plt.plot(X, all_results[i]["marg_likelihood"],
    #              label='{0} hidden units'.format(w))
    # ax.legend(numpoints=1, loc=3, frameon=False, prop={'size':'12'})
    # ax.set_ylabel('Marginal likelihood')

    # ax = fig.add_subplot(211)
    # for i, w in enumerate(widths):
    #     plt.plot(X, all_results[i]["tests_likelihood"])
    # ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    # ax.set_ylabel('Test Likelihood')
    # ax.set_xlabel('Training iteration')
    # #low, high = ax.get_ylim()
    # #ax.set_ylim([0, high])
    # fig.set_size_inches((5,3.5))
    # #ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    # plt.savefig('vary_widths.pdf', pad_inches=0.05, bbox_inches='tight')



def plot_traces_and_mean(all_results, trace_type, X=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    if X is None:
        X = np.arange(len(results[0][trace_type]))
    for i in xrange(N_samples):
        plt.plot(X, all_results[i][trace_type])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(trace_type)
    ax = fig.add_subplot(212)
    all_Y = [np.array(results[i][trace_type]) for i in range(N_samples)]
    plt.plot(X, sum(all_Y) / float(len(all_Y)))
    plt.savefig(trace_type + '.png')

if __name__ == '__main__':
    # results = run()
    # with open('results.pkl', 'w') as f:
    #     pickle.dump(results, f, 1)
    plot()
