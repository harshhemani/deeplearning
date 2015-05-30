import theano
import numpy as np
from theano import function
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class LogisticRegressionLayer:
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_bound = np.sqrt(6. / (n_in + n_out))
            initial_W = rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out))
            W_values = np.asarray(initial_W, dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        initial_W = rng.uniform(low=-W_bound, high=W_bound, size=filter_shape)
        initial_W = np.asarray(initial_W, dtype=theano.config.floatX)
        self.W = theano.shared( initial_W, borrow=True)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
