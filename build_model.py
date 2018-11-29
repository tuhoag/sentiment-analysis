import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    print(input_shape)
    inputs = Input(shape=input_shape)

    x = Dropout(rate=dropout_rate)(inputs)

    for _ in range(layers):
        x = Dense(units=units, activation='relu')(x)
        x = Dropout(rate=dropout_rate)(x)

    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=y)

    return model