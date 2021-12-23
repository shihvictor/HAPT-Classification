from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Conv1D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


def model(input_shape):
    input = Input(input_shape)

    X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation="relu")(input)
    X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation="relu")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)
    X = Dropout(rate=.2)(X)

    X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation="relu")(X)
    X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation="relu")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)
    X = Dropout(rate=.2)(X)

    X = Flatten()(X)
    X = Dense(units=64, activation="relu")(X)
    X = Dense(units=12, activation="relu")(X)

    model = Model(inputs=input, outputs=X)
    return model
