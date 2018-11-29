import tensorflow as tf

import vectorize_data
import build_model
import explore_data

def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):

    # load data
    (train_texts, train_labels), (val_texts, val_labels) = data
    num_classes = explore_data.get_num_classes(train_labels)

    # vectorize data
    x_train, x_val = vectorize_data.ngram_vectorize(train_texts, train_labels, val_texts)

    # build model
    model = build_model.mlp_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes
    )

    # train
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    history = model.fit(x=x_train,
        y=train_labels,
        validation_data=(x_val, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks)

    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))


