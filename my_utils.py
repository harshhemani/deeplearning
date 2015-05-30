import theano
from theano import function
from theano import tensor as T

import gzip
import numpy
import cPickle
import numpy as np

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                            dtype=theano.config.floatX),
                            borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                            dtype=theano.config.floatX),
                            borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset):
    f = gzip.open('data/'+dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
