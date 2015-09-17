from keras.layers.core import Activation, Dense
from keras_custom_layers import DAE
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import theano

nb_classes = 10
"""
# 96.06% accuracy on 50k-10k split # 430 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [5, 5, 5]
nb_finetune_epochs = 5
batch_sizes = [250, 300, 500]
"""

"""
# 94.51% accuracy on 50k-10k split # 177 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [0, 0, 0]
nb_finetune_epochs = 5
batch_sizes = [250, 300, 500]
"""

"""
# 96.96% accuracy on 50k-10k split # 500 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [0, 0, 0]
nb_finetune_epochs = 15
batch_sizes = [250, 300, 500]
"""

"""
# 97.35% accuracy on 50k-10k split # 575 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [1, 1, 1]
nb_finetune_epochs = 15
batch_sizes = [250, 300, 500]
"""
"""
# 97.37% accuracy on 50k-10k split # 534 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [1, 1, 1]
nb_finetune_epochs = 13
batch_sizes = [250, 300, 500]
"""

"""
# 97.18% accuracy on 50k-10k split # 522 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [2, 2, 2]
nb_finetune_epochs = 11
batch_sizes = [250, 300, 500]
"""

"""
# 97.20% accuracy on 50k-10k split # 548 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [3, 3, 3]
nb_finetune_epochs = 10
batch_sizes = [250, 300, 500]
"""

# 89.62% accuracy on 50k-10k split # 48 seconds
layer_sizes = [784, 450, 350, 100]
nb_pretrain_epochs = [0, 0, 0]
nb_finetune_epochs = 1
batch_sizes = [250, 300, 500]



#---------------------------------


"""
# 97.32% accuracy on 50k-10k split # 450 seconds
layer_sizes = [784, 400, 100]
nb_pretrain_epochs = [0, 0, 0]
nb_finetune_epochs = 25
batch_sizes = [250, 500, 1000]
"""


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

X_wholeset = np.vstack([X_train, X_test])
X_wholeset_tmp = X_wholeset

all_params = []
# do some pretraining, but keep saving parameters
print 'PRETRAINING'
for i in range(len(layer_sizes)-1):
    temp_ae_model = Sequential()
    temp_ae_model.add(DAE(layer_sizes[i], layer_sizes[i+1], activation='sigmoid'))
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    temp_ae_model.compile(loss='mean_squared_error', optimizer='adam')
    temp_ae_model.fit(X_wholeset_tmp, X_wholeset_tmp, nb_epoch=nb_pretrain_epochs[i], batch_size=batch_sizes[i])
    X_wholeset_tmp = temp_ae_model.predict(X_wholeset_tmp)
    W, b, bT = temp_ae_model.get_weights()
    all_params.append((W, b, bT))
# create model for fine tuning
final_ae_model = Sequential()
for i in range(len(layer_sizes)-1):
    dense_layer = Dense(layer_sizes[i], layer_sizes[i+1], activation='sigmoid')
    final_ae_model.add(dense_layer)
final_ae_model.add(Dense(layer_sizes[-1], nb_classes, activation='sigmoid'))
final_ae_model.add(Activation('softmax'))
final_ae_model.compile(loss='categorical_crossentropy', optimizer='adam')
# initialize weights
for i in range(len(layer_sizes)-1):
    W, b, bT = all_params[i]
    final_ae_model.layers[i].set_weights([W, b])
# finetune
print 'FINETUNING'
final_ae_model.fit(X_train, y_train, nb_epoch=nb_finetune_epochs, batch_size=200)
"""
print 'SAVING MODEL TO DISK'
final_ae_model.save_weights('mnist_ae_model.keras', overwrite=True)
"""
# evaluate performance
score = final_ae_model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print 'Test accuracy:', score[1]
