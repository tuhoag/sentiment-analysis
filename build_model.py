import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D
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

def sequence_model(inputs, pool_size, filters, blocks, kernel_size, embedding_dim, dropout_rate, num_words, input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Embedding(input_dim=num_words,
        output_dim=embedding_dim,
        input_length=input_shape[0])(inputs)

    for _ in range(blocks - 1):
        x = Dropout(rate=dropout_rate)(x)
        x = SeparableConv1D(filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            bias_initializer='random_uniform',
            depthwise_initializer='random_uniform',
            padding='same')(x)
        x = SeparableConv1D(filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            bias_initializer='random_uniform',
            depthwise_initializer='random_uniform',
            padding='same')(x)
        x = MaxPooling1D(pool_size=pool_size)(x)

    out_units, out_act = _calculate_model_output(num_classes)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(rate=dropout_rate)(x)
    y = Dense(units=out_units, activation=out_act)(x)

    model = Model(inputs=inputs, outputs=y)

    return model


