import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
# from PIL import Image
from decimal import *
from scipy.special import expit as sigmoid


class Data_mnist(object):
    '''Load data from a binary file. If it doesn't exist, it is created.'''
    def __init__(self, filename='norm_mnist.dat'):
        r_ts, r_xs, t_ts, t_xs = self.__load_mnist_data(filename)
        self.train_ts = r_ts
        self.train_xs = r_xs
        self.test_ts = t_ts
        self.test_xs = t_xs

    def __load_mnist_data(self, filename):
        ''' get training/testing data, preferably from binary file '''
        import os
        if os.path.isfile('./' + filename):
            with open(filename, 'rb') as f:
                t_train, x_train, t_test, x_test = pickle.load(f)
        else:
            print('No binary file. Generating ' + filename, end=' ')
            print('from mnist_train.csv and mnist_test.csv.')
            print('This is going to take a little while....')
            t_train, x_train, t_test, x_test = self.__clean_mnist_data(filename)
        return t_train, x_train, t_test, x_test

    def __clean_mnist_data(self, filename):
        '''Run this function the first time to organize data'''
        # Load training data file
        x_train = np.loadtxt('mnist_train.csv', delimiter=',', unpack=False)

        # Load test data file
        x_test = np.loadtxt('mnist_test.csv', delimiter=',', unpack=False)

        # Separate target values from data samples
        t_train = x_train[:, 0]
        t_train = t_train.astype(int)
        t_test = x_test[:, 0]
        t_test = t_test.astype(int)

        # Normalize the training/test data
        x_train = x_train / 255.0
        x_train[:, 0] = 1

        x_test = x_test / 255.0
        x_test[:, 0] = 1

        # Save data to binary file
        with open(filename, 'wb') as f:
            pickle.dump([t_train, x_train, t_test, x_test], f)

        return t_train, x_train, t_test, x_test


class NeuralNet(object):
    '''Two-layer neural network'''
    def __init__(self, num_input, num_hidden, num_output, eta=0.01, alpha=0.9,
                 target=0.9, num_epoch=50):

        self.n_xs = num_input
        self.n_h = num_hidden
        self.n_o = num_output
        self.n_epoch = num_epoch
        self.eta = eta
        self.alpha = alpha

        # Initialize weights to small random values
        self.wi, self.wj = self.__init_weights()

        # attribute to store previous weights in
        self.wi_0 = np.zeros(self.wi.shape)
        self.wj_0 = np.zeros(self.wj.shape)

        # target value used to compute error terms
        self.target = target

        # create attributes to store plotting data
        self.x_r = [i for i in range(num_epoch + 1)]
        self.y_r = [0]*(num_epoch + 1)
        self.x_t = [i for i in range(num_epoch + 1)]
        self.y_t = [0]*(num_epoch + 1)

    def __init_weights(self):
        num_weights = self.n_xs * self.n_h
        wi = np.random.uniform(-0.05, 0.05, num_weights)
        wi = wi.reshape(self.n_xs, self.n_h)

        num_weights = (self.n_h + 1) * self.n_o
        wj = np.random.uniform(-0.05, 0.05, num_weights)
        wj = wj.reshape(self.n_h + 1, self.n_o)
        return wi, wj

    def train(self, x, t, n_samp, X, T, N_samp):
        # Test the training and test data before training the neural net
        print('Epoch: 0\nTraining', end=' ')
        self.y_r[0] = self.test(x, t, n_samp)
        print('Test', end=' ')
        self.y_t[0] = self.test(X, T, N_samp)

        # Add an additional column of ones (bias) to the hidden nodes
        hid_j = np.ones(self.n_h+1)

        # Reformat the target values into arrays of 10 x n_samples
        target_mat = np.ones((self.n_o, self.n_o), float) - self.target
        np.fill_diagonal(target_mat, self.target)

        # Apply training algorithm for number of epochs
        for epoch in range(self.n_epoch):
            for i in range(n_samp):
                # Forward propagate
                hid_j, out_k = self.__forward(x[i, :], hid_j)

                # Back propagate
                # Calculate error terms
                error_o = out_k * (1 - out_k) * (target_mat[t[i]] - out_k)
                error_h = hid_j * (1 - hid_j) * np.dot(self.wj, error_o)

                # Update weights
                self.__update_weights(x[i, :], error_h, hid_j, error_o)

            # Test the training/test accuracy after each epoch
            print(' ')
            print('Epoch: ' + str(epoch+1))
            print('Training', end=' ')
            self.y_r[epoch+1] = self.test(x, t, n_samp)
            print('Test', end=' ')
            self.y_t[epoch+1] = self.test(X, T, N_samp)
        # After the last epoch, generate a confusion matrix
        self.conf_mat = self.__get_conf_mat(X, T, N_samp)
        return

    def __get_conf_mat(self, x, t, n_samp):
        '''generate confusion matrix'''
        hid_j = np.ones(self.n_h+1)
        conf_mat = np.zeros((10, 10), int)
        for i in range(n_samp):
            _, out_k = self.__forward(x[i, :], hid_j)
            a = t[i]
            b = np.argmax(out_k)
            conf_mat[a, b] += 1
        return conf_mat

    def test(self, x, t, n_samp):
        '''Test and compute the accuracy of the neural network'''
        hid_j = np.ones(self.n_h+1)
        n_correct = 0
        for i in range(n_samp):
            _, out_k = self.__forward(x[i, :], hid_j)
            if t[i] == np.argmax(out_k):
                n_correct += 1

        accuracy = 100.0 * n_correct / n_samp
        print('Accuracy = ' + str(accuracy) + '%')
        return accuracy

    def __forward(self, x, h):
        '''Forward propagate the inputs'''
        h[1:] = sigmoid(x.dot(self.wi))
        o = sigmoid(h.dot(self.wj))
        return h, o

    def __update_weights(self, x_i, error_j, hid_j, error_k):
        # Comupte delta in first layer
        delta_w_ji = (self.eta * np.outer(x_i, error_j[1:])) + (self.alpha * self.wi_0)
        # Update weights in first layer
        self.wi += delta_w_ji
        # Save the current delta for the next iteration
        self.wi_0 = delta_w_ji

        # Comupte delta in second layer
        delta_w_kj = (self.eta * np.outer(hid_j, error_k)) + (self.alpha * self.wj_0)
        # Update weights in second layer
        self.wj += delta_w_kj
        # Save the current delta for the next iteration
        self.wj_0 = delta_w_kj
        return


if __name__ == "__main__":
    getcontext().prec = 3
    start = time.clock()

    # get mnist data
    d = Data_mnist()
    n_train_samp, n_xs = d.train_xs.shape
    n_test_samp, _ = d.test_xs.shape

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # Experiment 1: Vary the number of hidden nodes
    epochs = 50
    n_hid_nodes = [20, 50, 100]
    momentum = 0.9
    eta = 0.1
    for nodes in n_hid_nodes:
        # Initialize neural net
        nn = NeuralNet(num_input=n_xs,
                       num_hidden=nodes,
                       num_output=10,
                       eta=eta,
                       alpha=momentum,
                       target=0.9,
                       num_epoch=epochs)

        # Start training neural net
        nn.train(d.train_xs, d.train_ts, n_train_samp,
                 d.test_xs, d.test_ts, n_test_samp)

        # Display elapsed time
        end = time.clock()
        print('Processing time:', Decimal(end) - Decimal(start), 'seconds')

        # Generate and save plot of results
        plt.plot(nn.x_r, nn.y_r, label='Training Data')
        plt.plot(nn.x_t, nn.y_t, label='Test Data')
        plt.ylim([0, 100])
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('CS 545: HW2 - Neural Networks - All Training Samples' +
                  '\nhidden layers = ' + str(nodes) +
                  ', momentum = ' + str(momentum) +
                  ', eta = ' + str(eta))

        save_file = ('nn_' + str(nodes) + '_hidLayers_' +
                     str(momentum) + '_momentum_allSamples.png')
        plt.savefig(save_file)

        plt.ylim([80, 100])
        save_file = ('nn_' + str(nodes) + '_hidLayers_' +
                     str(momentum) + '_momentum_allSamples_zoom.png')
        plt.savefig(save_file)
        plt.clf()

        print(save_file)
        print(nn.conf_mat)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # Experiment 2: Vary the momentum
    epochs = 50
    n_hid_nodes = 100
    momentum = [0, 0.25, 0.50]
    eta = 0.1
    for m in momentum:
        nn = NeuralNet(num_input=n_xs, num_hidden=n_hid_nodes,
                       num_output=10,
                       eta=eta,
                       alpha=m,
                       target=0.9,
                       num_epoch=epochs)

        nn.train(d.train_xs, d.train_ts, n_train_samp,
                 d.test_xs, d.test_ts, n_test_samp)

        end = time.clock()
        print('Processing time:', Decimal(end) - Decimal(start), 'seconds')

        # Generate and save plot of results
        plt.plot(nn.x_r, nn.y_r, label='Training Data')
        plt.plot(nn.x_t, nn.y_t, label='Test Data')
        plt.ylim([0, 100])
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('CS 545: HW2 - Neural Networks - All Training Samples' +
                  '\nhidden layers = ' + str(n_hid_nodes) +
                  ', momentum = ' + str(m) +
                  ', eta = ' + str(eta))

        save_file = ('nn_' + str(n_hid_nodes) + '_hidLayers_' +
                     str(m) + '_momentum_allSamples.png')
        plt.savefig(save_file)

        plt.ylim([80, 100])
        save_file = ('nn_' + str(n_hid_nodes) + '_hidLayers_' +
                     str(m) + '_momentum_allSamples_zoom.png')
        plt.savefig(save_file)
        plt.clf()

        print(save_file)
        print(nn.conf_mat)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # Experiment 3: Vary the momentum value
    # n_train_samp, n_xs = d.train_xs.shape

    batch_size_0p25 = int(n_train_samp/4)
    epochs = 50
    n_hid_nodes = 100
    momentum = 0.9
    eta = 0.1
    nn = NeuralNet(num_input=n_xs,
                   num_hidden=nodes,
                   num_output=10,
                   eta=eta,
                   alpha=momentum,
                   target=0.9,
                   num_epoch=epochs)

    nn.train(d.train_xs[0:batch_size_0p25], d.train_ts[0:batch_size_0p25],
             batch_size_0p25, d.test_xs, d.test_ts, n_test_samp)

    end = time.clock()
    print('Processing time:', Decimal(end) - Decimal(start), 'seconds')

    # Generate and save plot of results
    plt.plot(nn.x_r, nn.y_r, label='Training Data')
    plt.plot(nn.x_t, nn.y_t, label='Test Data')
    plt.ylim([0, 100])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('CS 545: HW2 - Neural Networks - One Quarter Training Samples'
              + '\nhidden layers = ' + str(nodes)
              + ', momentum = ' + str(momentum)
              + ', eta = ' + str(eta))

    save_file = ('nn_' + str(nodes) + '_hidLayers_' +
                 str(momentum) + '_momentum_quarterSamples.png')
    plt.savefig(save_file)

    plt.ylim([80, 100])
    save_file = ('nn_' + str(nodes) + '_hidLayers_' +
                 str(momentum) + '_momentum_quarterSamples_zoom.png')
    plt.savefig(save_file)
    plt.clf()

    print(save_file)
    print(nn.conf_mat)

    batch_size_0p50 = int(n_train_samp/2)
    nn = NeuralNet(num_input=n_xs,
                   num_hidden=nodes,
                   num_output=10,
                   eta=eta,
                   alpha=momentum,
                   target=0.9,
                   num_epoch=epochs)

    nn.train(d.train_xs[20000:20000+batch_size_0p50],
             d.train_ts[20000:20000+batch_size_0p50],
             batch_size_0p50,
             d.test_xs,
             d.test_ts,
             n_test_samp)

    end = time.clock()
    print('Processing time:', Decimal(end) - Decimal(start), 'seconds')

    # Generate and save plot of results
    plt.plot(nn.x_r, nn.y_r, label='Training Data')
    plt.plot(nn.x_t, nn.y_t, label='Test Data')
    plt.ylim([0, 100])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('CS 545: HW2 - Neural Networks - One Half Training Samples'
              + '\nhidden layers = ' + str(nodes)
              + ', momentum = ' + str(momentum)
              + ', eta = ' + str(eta))

    save_file = ('nn_' + str(nodes) + '_hidLayers_' +
                 str(momentum) + '_momentum_halfSamples.png')
    plt.savefig(save_file)

    plt.ylim([80, 100])
    save_file = ('nn_' + str(nodes) + '_hidLayers_' +
                 str(momentum) + '_momentum_halfSamples_zoom.png')
    plt.savefig(save_file)
    plt.clf()

    print(save_file)
    print(nn.conf_mat)
