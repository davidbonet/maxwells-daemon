from maxwell_d.data import load_data_subset
from cnn_utils import make_cnn_funs

# ------ Problem parameters -------
num_classes = 10
batch_size = 50
num_channels = 5
batch_size = 50
N_train = 10**3
N_tests = 10**3

# ------ Variational parameters -------
seed = 0

def run():
    (train_images, train_labels),\
    (tests_images, tests_labels) = load_data_subset(N_train, N_tests)
    conv_net, params, nllfun = make_cnn_funs(num_classes, num_channels, batch_size, seed)

if __name__ == '__main__':
    run()