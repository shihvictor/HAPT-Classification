from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Conv1D, Dense, Flatten, Activation, BatchNormalization

def model(input_shape):
    input = Input(input_shape)

    X = Dense(units=64, activation="relu")(input)
    X = Dense(units=12, activation='relu')(X)

    model = Model(inputs=input, outputs=X)
    return model


