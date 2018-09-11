import numpy as np
from sklearn.cross_validation import StratifiedKFold
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, merge, Convolution2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from data import select_data


def build_tetrode_model(tetrode_neuron):
    """
    Build tetrode convolutional neural network model for odor decoding.

    :param tetrode_neuron: (dictionary) mapping between tetrodes and neuron units
    :return: model: (keras) compiled decoding model
    """
    input_layers = []
    for t in tetrode_neuron.keys():
        k = tetrode_neuron[t]
        input_layers.append(Input(shape=(1, k + 1, 25)))
    convolution_layers = []
    for i, input_layer in enumerate(input_layers):
        t = tetrode_neuron.keys()[i]
        k = tetrode_neuron[t]
        convolution_layers.append(Convolution2D(5, k + 1, 1, activation='relu')(input_layer))
    combo = merge(convolution_layers, mode='concat', concat_axis=1)
    pooling = AveragePooling2D(pool_size=(1, 25))(combo)
    x = Flatten()(pooling)
    x = Dense(10, activation='relu')(x)
    x = Dropout(p=0.1)(x)
    prediction = Dense(4, activation='softmax')(x)
    model = Model(input_layers, prediction)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


def cross_validate(all_tetrode_data, tetrode_neuron, target, verbose=True):
    """
    Perform cross-validation with tetrode convolutional neural network model.

    :param all_tetrode_data: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
    :param tetrode_neuron: (dictionary) mapping between tetrodes and neuron units
    :param target: (2d numpy array) classification labels
    :param verbose: (bool) whether to print each validation fold accuracy
    :return: y_true, y_hat (2d numpy array) true and predicted labels
    """
    kf = StratifiedKFold(target.argmax(axis=-1), n_folds=10)
    y_true = np.zeros(target.shape)
    y_hat = np.zeros(target.shape)
    i = 0
    for train_index, test_index in kf:
        X_train, X_test = select_data(all_tetrode_data, train_index), select_data(all_tetrode_data, test_index)
        y_train, y_test = target[train_index, :], target[test_index, :]
        model = build_tetrode_model(tetrode_neuron)
        checkpointer = ModelCheckpoint('/home/linggel/neuroscience/temp_model.h5', verbose=0, save_best_only=True)
        hist = model.fit(X_train, y_train,
                         nb_epoch=10, batch_size=64,
                         validation_data=(X_test, y_test),
                         callbacks=[checkpointer], verbose=0)
        best_model = load_model('/home/linggel/neuroscience/current_model.h5')
        n = y_test.shape[0]
        y_true[i:(i + n), :] = y_test
        y_hat[i:(i + n), :] = best_model.predict(X_test)
        i += n
        if verbose:
            print('Current fold validation accuracy: {}'.format(max(hist.history['val_acc'])))
    return y_true, y_hat
