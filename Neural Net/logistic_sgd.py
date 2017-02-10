"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import pdb

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is the number of rows (N)
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]]
        # T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset, test_set, num_features):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here OPCC)
    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    # First column has ID numbers and can be excluded from the computation
    # Last column has the labels.
    data_file = numpy.genfromtxt(dataset, skip_header=1, delimiter=',', usecols = range(1, 95))
    test_data = numpy.genfromtxt(test_set, skip_header=1, delimiter=',', usecols = range(1, 95))
    
    train_target=data_file[0:34650, 93]
    test_target=test_data[:, 93]
    valid_target=data_file[34650:, 93]

    # Set index to second column
    train_target -= 1
    test_target -= 1
    valid_target -= 1

    data_file = load_information_gain_model(data_file, num_features)
    test_data = load_information_gain_model(test_data, num_features)

    #data_file = load_variance_model(data_file, num_features)
    #test_data = load_variance_model(test_data, num_features)

    #data_file = load_correlation_model(data_file, num_features)
    #test_data = load_correlation_model(test_data, num_features)
    
    train_set = (data_file[0:34650, 0:num_features], train_target)
    test_set = (test_data[:, 0:num_features], test_target)
    valid_set = (data_file[34650:, 0:num_features], valid_target)
    
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_data_pca(dataset, test_set, pca_set, pca_test, num_features):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here OPCC)
    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    # First column has ID numbers and can be excluded from the computation
    # Last column has the labels.
    data_file = numpy.genfromtxt(dataset, skip_header=1, delimiter=',', usecols = range(1, 95))
    test_data = numpy.genfromtxt(test_set, skip_header=1, delimiter=',', usecols = range(1, 95))
    
    train_target=data_file[0:34650, 93]
    test_target=test_data[:, 93]
    valid_target=data_file[34650:, 93]

    # Set index to first column
    train_target -= 1
    test_target -= 1
    valid_target -= 1

    data_file = numpy.genfromtxt(pca_set, delimiter=',')
    test_data = numpy.genfromtxt(pca_test, delimiter=',')
    
    train_set = (data_file[0:34650, 0:num_features], train_target)
    test_set = (test_data[:, 0:num_features], test_target)
    valid_set = (data_file[34650:, 0:num_features], valid_target)
    
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_information_gain_model(dataset, num_vars):
    """ Function that loads the columns based on order from highest to lowest information gain and the number of required columns.

    :type dataset: numPy array (dtype=float)
    :param dataset: a matrix of N rows and 93 columns

    :type num_vars: int
    :param num_vars: Number of features (columns) to select from the dataset.

    """
    information_gain_indices = numpy.array([10, 33, 25, 39, 59, 13, 14, 24, 85, 66, 41, 23, 35, 87, 68, 47, 61,
                                            3,  2, 63, 74,  7, 75, 67, 71, 89, 42, 31, 52,  8, 38, 15, 56, 26,
                                            32, 84, 69, 29, 53, 58, 16, 19, 34, 49, 70, 37, 79,  0, 77, 40, 91,
                                            1, 86, 55, 72, 46, 65, 21, 82, 17, 90, 36, 78, 57, 12, 43, 54, 73,
                                            28, 88, 81, 18,  9, 20, 44, 64, 22, 62, 76, 51, 45,  6, 48, 11, 60,
                                            92, 30,  4, 27, 80, 50, 5, 83], dtype=numpy.int64)
    
    information_gain_indices = information_gain_indices[num_vars:]
    print(information_gain_indices.shape[0])

    dataset = numpy.delete(dataset, information_gain_indices, 1)

    #pdb.set_trace()    
    return dataset

def load_variance_model(dataset, num_vars):
    """ Function that loads the columns based on order from highest to lowest variance and the number of required columns.

    :type dataset: numPy array (dtype=float)
    :param dataset: a matrix of N rows and 93 columns

    :type num_vars: int
    :param num_vars: Number of features (columns) to select from the dataset.

    """

    variance_indices = numpy.array([72, 66, 18, 89, 23, 57, 68, 53, 14, 73,  8, 63, 33, 39, 34, 38, 74,
                                    13, 10, 59,  2, 26, 75,  3, 77, 85, 61, 58, 47, 35, 44, 71, 82, 45,
                                    24, 31, 79, 15,  7, 87, 90, 12, 55, 41, 69, 52, 84, 28, 49, 67, 25,
                                    42, 29, 88, 86, 37, 16,  0, 21, 19, 46, 48, 43, 65, 17, 32, 78, 92,
                                    1, 83, 70, 40,  9, 56, 51, 60,  6, 54, 91, 62, 76, 30, 81, 27, 64,
                                    36, 22, 20, 11, 50,  4, 80, 5], dtype=numpy.int64)
    variance_indices = variance_indices[num_vars:]
    print(variance_indices.shape[0])


    dataset = numpy.delete(dataset, variance_indices, 1)
    #pdb.set_trace() 
    return dataset

def load_correlation_model(dataset, num_vars):
    """ Function that loads the columns based on order from highest to lowest correlation and the number of required columns.

    :type dataset: numPy array (dtype=float)
    :param dataset: a matrix of N rows and 93 columns

    :type num_vars: int
    :param num_vars: Number of features (columns) to select from the dataset.

    """
    correlation_indices = numpy.array([13, 39, 24, 14, 87, 23, 35, 19, 68, 71,  7, 74, 21, 37, 40, 75,  1,
                                    17, 89, 66, 12, 10, 61, 67, 65, 32, 53,  8, 57, 82, 78, 54, 56, 58,
                                    3, 59, 79,  2, 34, 90, 91, 25,  6, 46, 27, 63, 48, 43, 70, 36, 18,
                                    16, 41, 28, 42, 81, 45, 52, 60, 85, 88, 22, 30, 77, 26, 84, 86,  9,
                                    38, 80, 49, 47, 11, 51, 72,  0, 62, 73, 44, 64, 50, 76, 92,  5,  4,
                                    69, 55, 29, 33, 31, 20, 15, 83], dtype=numpy.int64)
    correlation_indices = correlation_indices[num_vars:]
    print(correlation_indices.shape[0])


    dataset = numpy.delete(dataset, correlation_indices, 1)
    #pdb.set_trace() 
    return dataset

def sgd_optimization_opcc(learning_rate=0.05, n_epochs=1000,
                           dataset='OPCCTrain.csv',
                           test_set = 'OPCCTest.csv',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on the OPCC dataset.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the csv filename

    """
    #datasets = load_data(dataset, test_set, 68)
    #datasets = load_data(dataset, test_set, 86)
    #datasets = load_data(dataset, test_set, 92)
    #datasets = load_data(dataset, test_set, 93)

    #PCA
    #datasets = load_data_pca(dataset, test_set, 'xTrain_PCA_68.csv', 'xTest_PCA_68.csv', 68)
    #datasets = load_data_pca(dataset, test_set, 'xTrain_PCA_86.csv', 'xTest_PCA_86.csv', 86)
    datasets = load_data_pca(dataset, test_set, 'xTrain_PCA_92.csv', 'xTest_PCA_92.csv', 92)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #Calculate D - dimension
    num_features = train_set_x.shape[1].eval()

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # Each OPCC product has size 93 features
    classifier = LogisticRegression(input=x, n_in=num_features, n_out=9)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 3  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl', 'rb'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='OPCCTrain.csv'
    test_set='OPCCTest.csv'
    #datasets = load_data(dataset, test_set, 68)
    datasets = load_data(dataset, test_set, 86)
    #datasets = load_data(dataset, test_set, 92)
    #datasets = load_data(dataset, test_set, 93)

    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()

    print(test_set_y[:10])
    predicted_values = predict_model(test_set_x)
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values[:10])

    errors = 0
    for i in range(len(test_set_y)):
        if predicted_values[i] != test_set_y[i]:
            errors = errors+1

    error_rate = (errors / len(test_set_y)) * 100
    print(error_rate)

    #print(predicted_values)


if __name__ == '__main__':
    sgd_optimization_opcc()
    #predict()
