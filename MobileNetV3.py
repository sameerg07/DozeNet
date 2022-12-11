"""MobileNet v3 small models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""
# print(1)
# exit(0)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Dropout
from tensorflow.keras.utils import plot_model


"""MobileNet v3 models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

import tensorflow
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape

from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from datetime import date, datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from scipy.stats import norm
from tensorflow.keras.optimizers import Adam,Adagrad
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class MobileNetBase:
    def __init__(self, shape, n_class, alpha=1.0):
        """Init
        
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
        """
        self.shape = shape
        self.n_class = n_class
        self.alpha = alpha

    def _relu6(self, x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0
    
    def _gelu(self,x):
        """Gelu
        """
        return tensorflow.keras.activations.gelu(x, approximate=False)

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.

        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)
        if nl == 'GE':
            x = Activation(self._gelu)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        # x = Dropout(0.5)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x
    
    def _squeeze_gelu(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='gelu')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        # x = Dropout(0.5)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        # x = Dropout(0.5)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def _bottleneck_gelu(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        # x = Dropout(0.5)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze_gelu(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        # x = Dropout(0.5)(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass

def define_model():
    """Defines CNN layers"""
    # initialise model
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv1D(filters=20, kernel_size=125, input_shape=(7500, 1)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=10))
    model.add(Dropout(0.3))

    # Convolutional Layer 2
    model.add(Conv1D(filters=40, kernel_size=50))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.3))

    # Convolutional Layer 3
    model.add(Conv1D(filters=60, kernel_size=10))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.3))

    # Fully Connected Layer 1
    model.add(Flatten())
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.3))

    # Fully Connected Layer 2
    model.add(Dense(2, activation='softmax'))

    # configure model for training
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model

class MobileNetV3_Large(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.
        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Large, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Large.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        # x = Dropout(0.5)(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            # x = Dropout(0.5)(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)
        
        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)

        return model

class MobileNetV3_LargeGELU(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.
        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_LargeGELU, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Large.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='GE')

        x = self._bottleneck_gelu(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck_gelu(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = self._bottleneck_gelu(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = self._bottleneck_gelu(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck_gelu(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck_gelu(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck_gelu(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='GE')
        x = self._bottleneck_gelu(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='GE')
        x = self._bottleneck_gelu(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='GE')
        x = self._bottleneck_gelu(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='GE')
        x = self._bottleneck_gelu(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='GE')

        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='GE')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        # x = Dropout(0.5)(x)
        x = self._return_activation(x, 'GE')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            # x = Dropout(0.5)(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)
        
        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)

        return model


class MobileNetV3_Small(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.

        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Small, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        # x = Dropout(0.5)(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            # x = Dropout(0.5)(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        if plot:
            plot_model(model, to_file='MobileNetv3_small.png', show_shapes=True)


        return model


class MobileNetV3_SmallGELU(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.

        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_SmallGELU, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='GE')

        x = self._bottleneck_gelu(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck_gelu(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck_gelu(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck_gelu(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='GE')
        x = self._bottleneck_gelu(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='GE')

        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='GE')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        # x = Dropout(0.5)(x)
        x = self._return_activation(x, 'GE')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            # x = Dropout(0.5)(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        if plot:
            plot_model(model, to_file='MobileNetv3_small.png', show_shapes=True)


        return model


def visualise_training(history,prefix):
    print(history.history)
    # access the accuracy and loss values found throughout training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    

    fig = plt.figure(figsize=(10,5))

    epochs = range(1, len(acc) + 1)
    # plot accuracy throughout training (validation and training)
    plt.plot(epochs, acc, color='xkcd:azure')
    plt.plot(epochs, val_acc, color='xkcd:green')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc='upper left')
    fig = plt.figure(figsize=(10,5))

    # plot loss throughout training (validation and training)
    plt.plot(epochs, loss, color='xkcd:azure')
    plt.plot(epochs, val_loss, color='xkcd:green')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc='upper left')
    fig = plt.figure(figsize=(10,5))
    

    # plot accuracy throughout training (just training)
    plt.plot(epochs, acc, 'b')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    fig = plt.figure(figsize=(10,5))

    # plot loss throughout training (just training)
    plt.plot(epochs, loss, 'b')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig = plt.figure(figsize=(10,5))
    fig.savefig(prefix+'.jpg', bbox_inches='tight', dpi=150)
    # plt.show()



def evaluate_model(accuracy, y_test, y_pred):
    # calculate accuracy as a percentage
    accuracy = accuracy * 100.0
    print('Accuracy =', accuracy, "%")

    # generate confusion matrix
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print('Confusion Matrix:')
    print(np.matrix(matrix))

    # calculate d'
    tp, fn, fp, tn = matrix.ravel()
    dprime = norm.ppf(tn/(tn+fp)) - norm.ppf(fn/(tp+fn))
    print('dPrime =', dprime)

    # generate classification report
    target_names = ['non-apnea', 'apnea']
    print('Classification Report:')
    report = classification_report(y_test.argmax(axis=1),
          y_pred.argmax(axis=1), target_names=target_names)
    print(report)
    return report,np.matrix(matrix)


ip_shape = (1,7500,1)
n_classes = 2
lr = None
epsilon = None
modelName = "mobnet2Dv31SmallGELU"
logspath = "./ModelLogs/"+modelName


model = MobileNetV3_SmallGELU(ip_shape,n_classes,alpha=1.0,include_top=True).build(plot=False)
# model = define_model()
print(model.summary())

with open("ModelReports/"+modelName+"_overall.txt",'a') as f:
    f.write("ip_shape"+str(ip_shape))
    f.write("lr,epsilon: "+str(lr)+","+str(epsilon))
    # f.write(str(model.summary()))
    f.write("\n")
    f.close()


X = np.load("../../X.npy")
Y = np.load("../../Y.npy")
print("Orginal shape of X: ",X.shape)
print("Orginal shape of Y: ",Y.shape)
# reshape input of (7500,1) to (125,60,1)
X = X.reshape(X.shape[0],ip_shape[0],ip_shape[1],ip_shape[2])

# change targets to one hot encoded form
Y = to_categorical(Y)
Y = Y.reshape(-1, 2)
print("Latest shape of Y: ",Y.shape)
print(Y)

# shuffle data before split
x_shuffle, y_shuffle = shuffle(X, Y, random_state=2)

# split inputs and targets into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_shuffle, y_shuffle, test_size=0.1)

print("Data Ready")

verbose, epochs, batch_size = 1, 100, 128

with open("ModelReports/"+modelName+"_overall.txt",'a') as f:
    f.write(str((verbose, epochs, batch_size)))
    f.write("\n")
    f.close()

# initialise early stopping callback
es = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                    verbose=1, restore_best_weights=True)
# opt = Adam(lr=lr,epsilon = epsilon)
# opt = Adagrad(lr=lr,epsilon = epsilon)
model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])


if not os.path.exists(logspath):
    os.mkdir(logspath)

tb_callback = tensorflow.keras.callbacks.TensorBoard(
    log_dir=logspath,
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",
)  # How 
# train model
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    verbose=verbose, validation_split=0.2, callbacks=[es,tb_callback])


model.save("models/"+modelName)
# model.save_weights('models/mobnet2Dv1.h5')
df = pd.DataFrame.from_dict(history.history)
df.to_csv('models/'+modelName+'hist.csv', encoding='utf-8', index=False)


# test model and return accuracy
_, accuracy = model.evaluate(
    x_test, y_test, batch_size=batch_size, verbose=0)

# find predictions model makes for test set
y_pred = model.predict(x_test)

report,confmatrix = evaluate_model(accuracy, y_test, y_pred)
with open("ModelReports/"+modelName+"_overall.txt",'a') as f:
    f.write(str("Classificatin Report: \n"+report))
    f.write(str("ConfMatrix: \n")+str(confmatrix))
    f.write("\n")
    f.close()
visualise_training(history,modelName)
