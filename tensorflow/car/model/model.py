from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, SimpleRNN, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2


def cnn3_full1():
    img_input = Input(shape=(120, 160, 3), name="img_input")

    x = Convolution2D(8, 3, 3)(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(16, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(32, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    merged = Flatten()(x)

    x = Dense(256)(merged)
    x = Activation('linear')(x)
    x = Dropout(.2)(x)

    angle_out = Dense(1, name="angle_out")(x)

    model = Model(input=[img_input], output=[angle_out])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def cnn3_full1_rnn1():
    img_input = Input(shape=(120, 160, 3), name="img_input")

    x = Convolution2D(8, 3, 3)(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(16, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(32, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    merged = Flatten()(x)

    x = Dense(256)(merged)
    x = Activation('linear')(x)
    x = Dropout(.2)(x)

    x = Reshape((1, 256))(merged)
    x = SimpleRNN(256, activation='linear')(x)

    throttle_out = Dense(1, name="throttle_out")(x)
    angle_out = Dense(1, name="angle_out")(x)

    model = Model(input=[img_input], output=[angle_out])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def cnn1_full1():
    img_input = Input(shape=(120, 160, 3), name="img_input")

    x = Convolution2D(1, 3, 3)(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    merged = Flatten()(x)

    x = Dense(32)(merged)
    x = Activation('linear')(x)
    x = Dropout(.05)(x)

    angle_out = Dense(1, name="angle_out")(x)

    model = Model(input=[img_input], output=[angle_out])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def norm_cnn3_full1():
    img_input = Input(shape=(120, 160, 3), name="img_input")

    x = BatchNormalization()(img_input)
    x = Convolution2D(8, 3, 3)(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(16, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(32, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    merged = Flatten()(x)

    x = Dense(256)(merged)
    x = Activation('linear')(x)
    x = Dropout(.2)(x)

    angle_out = Dense(1, name="angle_out")(x)

    model = Model(input=[img_input], output=[angle_out])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def vision_2D(dropout_frac=.2):
    '''
    Network with 4 convolutions, 2 residual shortcuts to predict angle.
    '''
    img_input = Input(shape=(120, 160, 3), name="img_input")

    net = Convolution2D(64, 6, 6, subsample=(4, 4), name='conv0')(img_input)
    net = dropout_frac(dropout_frac)(net)

    net = Convolution2D(64, 3, 3, subsample=(2, 2), name='conv1')(net)
    net = dropout_frac(dropout_frac)(net)

    # Create residual to shortcut
    aux1 = Flatten(name='aux1_flat')(net)
    aux1 = Dense(64, name='aux1_dense')(aux1)

    net = Convolution2D(128, 6, 6, subsample=(
        2, 2), border_mode='same', name='conv2')(net)
    net = dropout_frac(dropout_frac)(net)

    net = Convolution2D(128, 3, 3, subsample=(
        2, 2), border_mode='same', name='conv3')(net)
    net = dropout_frac(dropout_frac)(net)

    aux2 = Flatten(name='aux2_flat')(net)
    aux2 = Dense(64, name='aux2_dense')(aux2)

    net = Flatten(name='net_flat')(net)
    net = Dense(512, activation='relu', name='net_dense1')(net)
    net = Dropout(dropout_frac)(net)
    net = Dense(256, activation='relu', name='net_dense2')(net)
    net = Dropout(dropout_frac)(net)
    net = Dense(128, activation='relu', name='net_dense3')(net)
    net = Dropout(dropout_frac)(net)
    net = Dense(64, activation='relu', name='net_dense4')(net)

    # combine rsidual layers
    net = merge([net, aux1, aux2], mode='sum')
    angle_out = Dense(1, name='angle_out')(net)
    model = Model(input=[img_input], output=[angle_out])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def regularized_cnn4_full1():
    reg = l2(0.005)

    img_input = Input(shape=(120, 160, 3), name='img_input')

    x = Convolution2D(4, 3, 3, W_regularizer=reg)(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(8, 3, 3, W_regularizer=reg)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(16, 3, 3, W_regularizer=reg)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(32, 3, 3, W_regularizer=reg)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(128, W_regularizer=reg)(x)
    x = Activation('linear')(x)
    x = Dropout(.2)(x)

    angle_out = Dense(1, name='angle_out')(x)

    model = Model(input=[img_input], output=[angle_out])
    model.compile(optimizer='adam', loss='mean_sequared_error')
    return model
