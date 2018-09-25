import numpy as np


def stack_data(all_data, window_size, stride):
    """
    Divide data into smaller time windows with stride and stack them.

    :param all_data: (3d numpy array) spike train or LFP data of format [trial, neuron, time]
    :param window_size: (int) width of time window
    :param stride: (int) as stride in convolution
    :return: stacked_data: (4d numpy array) data of format [trial, window, neuron, time]
    """
    n_trial, n_neuron, n_time = all_data.shape
    n_window = (n_time - window_size) / stride
    stacked_data = np.zeros((n_trial, n_window, n_neuron, window_size))
    for i in range(n_window):
        window_start = i * stride
        window_end = i * stride + window_size
        stacked_data[:, i, :, :] = all_data[:, :, window_start:window_end]
    return stacked_data


def organize_tetrode(spike_data, lfp_data, tetrode_ids, tetrode_units, verbose=True):
    """
    Organize spike and LFP data by tetrode.

    :param spike_data: (3d numpy array) spike train data of format [trial, neuron, time]
    :param lfp_data: (3d numpy array ) LFP data of format [trial, tetrode, time]
    :param tetrode_units: (dictionary) mapping between tetrodes and neuron units
    :param verbose: (bool) whether to print each tetrode data shape
    :return: all_tetrode_data: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
    """
    all_tetrode_data = []
    i = 0
    for j, t in enumerate(tetrode_ids):
        k = tetrode_units[t]
        if k == 0:
            continue
        tetrode_lfp = np.expand_dims(lfp_data[:, j, :], axis=1)
        tetrode_spike = spike_data[:, i:(i + k), :]
        if len(tetrode_spike.shape) == 2:
            tetrode_spike = np.expand_dims(tetrode_spike, axis=1)
        tetrode_data = np.concatenate([tetrode_lfp, tetrode_spike], axis=1)
        tetrode_data = np.expand_dims(tetrode_data, axis=1)
        all_tetrode_data.append(tetrode_data)
        if verbose:
            print('Current tetrode {t} with {k} neurons/units'.format(t=t, k=k))
            print(tetrode_data.shape)
        i += k
    return all_tetrode_data


def select_data(all_tetrode_data, index):
    """
    Select tetrode data by trial indices.

    :param all_tetrode_data: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
    :param index: (1d numpy array) trial indices
    :return: current_data: (list of 4d numpy arrays) selected subset of tetrode data
    """
    current_data = []
    for x in all_tetrode_data:
        current_data.append(x[index, :, :, :])
    return current_data


def valid_tetrodes(tetrode_ids, tetrode_units):
    return [x for x in tetrode_ids if tetrode_units[x] > 0]
