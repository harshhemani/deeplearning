import theano
import numpy as np
import theano.tensor as T
from keras.layers.core import Layer, Activation
from keras.utils.theano_utils import shared_zeros
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from keras import initializations, activations, regularizers, constraints

class DAE(Layer):
    '''
        Denoising AutoEncoder
    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None, name=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, corruption_level=0.0):

        super(DAE, self).__init__()
        self.srng = RandomStreams(seed=np.random.randint(10e6))
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.corruption_level = corruption_level

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.bT = shared_zeros((self.input_dim))

        self.params = [self.W, self.b, self.bT]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        self.b.name = '%s_b' % name
        self.bT.name = '%s_bT' % name

    def get_output(self, train=False):
        X = self.get_input(train)
        if train:
            X *= self.srng.normal(size=X.shape, avg=1.0, std=T.sqrt(self.corruption_level / (1.0 - self.corruption_level)), dtype=theano.config.floatX) 
            output = self.activation(T.dot(self.activation(T.dot(X, self.W) + self.b), self.W.T) + self.bT)
        else:
            output = self.activation(T.dot(X, self.W) + self.b)
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                "corruption_level": self.corruption_level}
