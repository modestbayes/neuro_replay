import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

lfp_tetrode = [19, 4, 21, 20, 8, 9, 16, 1, 2, 23, 22, 7, 13, 14, 15] # list of tetrodes
tetrode_neuron = {1:4, 2:1, 4:1, 7:2, 8:4, 9:1, 13:4, 14:4, 15:5, 16:3, 19:9, 20:2, 21:6, 22:4, 23:4} # dictionary mapping between tetrodes and neuron units

def tetrode_data(spike_data, lfp_data, verbose=False):
    """
    Organize spike and lfp data by tetrode.
    
    Args
        spike_data: (3d numpy array) [trial, neuron, time]
        lfp_data: (3d numpy array) [trial, channel, time]
        
    Returns
        all_tetrode_data: (list) of 4d numpy array [trial, 1, combined dimension, time]
    """
    all_tetrode_data = []
    i = 0
    for t in sorted(lfp_tetrode):
        j = lfp_tetrode.index(t)
        tetrode_lfp = lfp_data[:, j, :].reshape((194, 1, 25))
        k = tetrode_neuron[t]
        tetrode_spike = spike_data[:, i:(i + k), :].reshape((194, k, 25))
        tetrode_data = np.concatenate([tetrode_lfp, tetrode_spike], axis=1)
        tetrode_data = np.expand_dims(tetrode_data, axis=1)
        if verbose:
            print('{} neuron/units'.format(k))
            print('Current tetrode {}'.format(t))
            print(tetrode_data.shape)
        all_tetrode_data.append(tetrode_data)
        i += k
    return(all_tetrode_data)


def build_tetrode_model():
    """
    Build tetrode-wise convolution model.
    """
    input_layers = []
    for t in sorted(lfp_tetrode):
        k = tetrode_neuron[t]
        input_layers.append(Input(shape=(1, k + 1, 25)))
    convolution_layers = []
    for i, input_layer in enumerate(input_layers):
        t = sorted(lfp_tetrode)[i]
        k = tetrode_neuron[t]
        convolution_layers.append(Convolution2D(5, k + 1, 1, activation='relu')(input_layer))
    combo = merge(convolution_layers, mode='concat', concat_axis=1)
    pooling = AveragePooling2D(pool_size=(1, 25))(combo)
    x = Flatten()(pooling)
    x = Dense(10, activation='relu')(x)
    x = Dropout(p=0.1)(x)
    prediction = Dense(5, activation='softmax')(x)
    model = Model(input_layers, prediction)
    return(model)