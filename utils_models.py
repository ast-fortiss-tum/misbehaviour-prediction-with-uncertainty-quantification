from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2

from utils import INPUT_SHAPE


def build_model(model_name, rate, use_dropout=False):
    """
    Retrieve the DAVE-2 NVIDIA model
    """
    model = None
    if "dave2" in model_name:
        model = create_dave2_model(rate,use_dropout)
    else:
        print("Incorrect model name provided")
        exit()

    assert model is not None
    model.summary()

    return model


def create_dave2_model(drop_rate, use_dropout=False):
    """
    Modified NVIDIA model w/ Dropout layers
    """
    if use_dropout:
        rate = drop_rate
        inputs = keras.Input(shape=INPUT_SHAPE)
        lambda_layer = keras.layers.Lambda(lambda x: x / 127.5 - 1.0, name="lambda_layer")(inputs)
        x = keras.layers.Conv2D(24, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(
            lambda_layer)
        x = keras.layers.Dropout(rate)(x, training=True)
        x = keras.layers.Conv2D(36, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate)(x, training=True)
        x = keras.layers.Conv2D(48, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate)(x, training=True)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate)(x, training=True)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate)(x, training=True)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate)(x, training=True)
        x = keras.layers.Dense(50, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate)(x, training=True)
        x = keras.layers.Dense(10, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate)(x, training=True)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    else:
        """
        Dave2 model without dropout layers (for Deep Ensembles)
        """
        inputs = keras.Input(shape=INPUT_SHAPE)
        lambda_layer = keras.layers.Lambda(lambda x: x / 127.5 - 1.0, name="lambda_layer")(inputs)
        x = keras.layers.Conv2D(24, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(
            lambda_layer)
        x = keras.layers.Conv2D(36, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Conv2D(48, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dense(50, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dense(10, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

    return model
