import train_ngram_model

def tune_ngram_model(data):
    num_layers = [1, 2, 3]
    num_units = [8, 16, 32, 64, 128]

    params = {
        'layers': [],
        'units': [],
        'accuracy': [],
    }

    for layers in num_layers:
        for units in num_units:
            params['layers'].append(layers)
            params['units'].append(units)

            _, val_acc = train_ngram_model.train_ngram_model(data, units=units, layers=layers)

            params['accuracy'].append(val_acc)

    plot_parameters(params)

def plot_parameters(params):
    pass
