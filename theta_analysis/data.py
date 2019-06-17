import numpy as np


def load_data(rat_name):
    """
    Load processed Numpy data arrays.
    """
    trial_info = np.load('/Users/linggeli/neuroscience/data/{}_trial_info.npy'.format(rat_name))
    spike_data = np.load('/Users/linggeli/neuroscience/data/{}_spike_data_binned.npy'.format(rat_name))
    lfp_data = np.load('/Users/linggeli/neuroscience/data/{}_lfp_data_sampled.npy'.format(rat_name))
    lfp_data = np.swapaxes(lfp_data, 1, 2)
    return trial_info, spike_data, lfp_data


def filter_trials(trial_info):
    """
    Get indices of correct in-sequence trials of odors A to D.
    """
    rat_correct = trial_info[:, 0] == 1
    in_sequence = trial_info[:, 1] == 1
    not_odor_e = trial_info[:, 3] < 5
    select = rat_correct & in_sequence & not_odor_e
    return select


def clean_data(trial_info, spike_data, lfp_data):
    """
    Clean up trials, remove cells that do not fire, and label targets.
    """
    trial_indices = filter_trials(trial_info)
    spike_data = spike_data[trial_indices]
    lfp_data = lfp_data[trial_indices]
    total_spike = np.sum(np.sum(spike_data[:, :, 200:300], axis=2), axis=0)
    spike_data = spike_data[:, total_spike > 0, :]
    target = trial_info[trial_indices, 3] - 1
    return target, spike_data, lfp_data


def reference_tetrode(rat_name):
    """
    Return the index of reference tetrode.
    """
    if rat_name == 'superchris':
        tetrode_index = 3
    if rat_name == 'stella':
        tetrode_index = 3
    if rat_name == 'barat':
        tetrode_index = 2
    if rat_name == 'buchanan':
        tetrode_index = 11
    if rat_name == 'mitt':
        tetrode_index = 2
    return tetrode_index
