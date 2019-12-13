import numpy as np


def filter_trials(trial_info):
    rat_correct = trial_info[:, 0] == 1
    in_sequence = trial_info[:, 1] == 1
    not_odor_e = trial_info[:, 3] < 5
    select = rat_correct & in_sequence & not_odor_e
    return select


def prepare_data(rat_name):
    # load data
    spike_data_binned = np.load('/home/linggel/neuroscience/processed_data/{}_spike_data_binned.npy'.format(rat_name))
    lfp_data_sampled = np.load('/home/linggel/neuroscience/processed_data/{}_lfp_data_sampled.npy'.format(rat_name))
    lfp_data_sampled = np.swapaxes(lfp_data_sampled, 1, 2)
    trial_info = np.load('/home/linggel/neuroscience/processed_data/{}_trial_info.npy'.format(rat_name))
    # process data
    trial_indices = filter_trials(trial_info)
    decoding_start = 210
    decoding_end = decoding_start + 25
    # scale data first
    spike_data_binned = spike_data_binned[trial_indices, :, :]
    spike_data_binned = (spike_data_binned - np.mean(spike_data_binned)) / np.std(spike_data_binned)
    lfp_data_sampled = lfp_data_sampled[trial_indices, :, :]
    decoding_data_spike = spike_data_binned[:, :, decoding_start:decoding_end]
    decoding_data_lfp = lfp_data_sampled[:, :, decoding_start:decoding_end]
    decoding_target = np_utils.to_categorical((trial_info[trial_indices, 3] - 1).astype(int))
    # organize tetrode data
    if rat_name == 'superchris':
        tetrode_ids = [1, 10, 12, 13, 14, 15, 16, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {1:3, 10:0, 12:1, 13:8, 14:4, 15:6, 16:1, 18:0, 19:4, 2:3, 
                         20:0, 21:1, 22:5, 23:7, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1}
    elif rat_name == 'stella':
        tetrode_ids = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {10:1, 12:0, 13:5, 14:7, 15:4, 16:8, 17:0, 18:0, 19:1, 20:0, 
                     21:1, 22:1, 23:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:13, 8:4, 9:4}
    elif rat_name == 'buchanan':
        tetrode_ids = [10, 12, 13, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 2, 4, 5, 6, 7, 8, 9]
        tetrode_units = {10:0, 12:0, 13:9, 15:6, 16:0, 17:4, 18:12, 19:15, 1:0, 
                     20:0, 21:1, 22:13, 23:8, 2:2, 4:0, 5:6, 6:3, 7:0, 8:0, 9:0}
    elif rat_name == 'barat':
        tetrode_ids = [10, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {10:1, 12:20, 13:12, 14:7, 15:0, 16:0, 17:11, 18:0, 19:0, 1:1, 20:0, 21:9, 22:0, 
                         23:1, 2:0, 3:11, 4:0, 5:4, 6:0, 7:1, 8:0, 9:14}
    elif rat_name == 'mitt':
        tetrode_ids = [12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {12:16, 13:15, 14:4, 15:2, 16:6, 17:2, 18:12, 19:15, 1:0, 20:12, 21:0, 22:1, 
                         23:4, 2:0, 3:4, 4:0, 5:0, 6:0, 7:3, 8:1, 9:7}
    tetrode_data = organize_tetrode(decoding_data_spike, decoding_data_lfp, tetrode_ids, tetrode_units)
    return tetrode_data, decoding_target, tetrode_ids, tetrode_units, spike_data_binned, lfp_data_sampled


def stack_data(all_data, window_size, stride):
    """
    Divide data into smaller time windows with stride and stack them.

    :param all_data: (3d numpy array) spike train or LFP data of format [trial, neuron, time]
    :param window_size: (int) width of time window
    :param stride: (int) as stride in convolution
    :return: (4d numpy array) data of format [trial, window, neuron, time]
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
    :param tetrode_ids: (list) of tetrode ids in the order of LFP data
    :param tetrode_units: (dict) number of neuron units on each tetrode
    :param verbose: (bool) whether to print each tetrode data shape
    :return: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
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
        tetrode_data = np.expand_dims(tetrode_data, axis=-1)

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
    :return: (list of 4d numpy arrays) selected subset of tetrode data
    """
    current_data = []
    for x in all_tetrode_data:
        current_data.append(x[index, :, :, :])
    return current_data


def valid_tetrodes(tetrode_ids, tetrode_units):
    """
    Only keep valid tetrodes with neuron units so that there is corresponding spike train data.

    :param tetrode_ids: (list) of tetrode ids in the order of LFP data
    :param tetrode_units: (dict) number of neuron units on each tetrode
    :return: (list) of tetrode ids with neuron units
    """
    return [x for x in tetrode_ids if tetrode_units[x] > 0]
