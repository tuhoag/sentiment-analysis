import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def _calculate_model_output(num_classes):
    if num_classes == 2:
        out_units = 1
        out_act = 'sigmoid'
    elif num_classes > 2:
        out_units = num_classes
        out_act = 'softmax'
    else:
        out_units = 0
        out_act = None

    return out_units, out_act

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Dropout(rate=dropout_rate)(inputs)

    for _ in range(layers):
        x = Dense(units=units, activation='relu')(x)
        x = Dropout(rate=dropout_rate)(x)

    out_units, out_act = _calculate_model_output(num_classes)
    y = Dense(out_units, activation=out_act)(x)

    model = Model(inputs=inputs, outputs=y)

    return model