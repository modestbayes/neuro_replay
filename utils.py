import numpy as np
from data import stack_data, organize_tetrode


def predict_all(model, all_data):
    """
    Predict odor probabilities for all trials.

    :param model: (keras) decoding model
    :param all_data: (4d numpy array) data of format [trial, window, neuron, time]
    :return: all_pred: (3d numpy array) prediction of format [trial, time, odor]
    """
    test = stack_data(all_data, 25, 10)
    n_trial, n_window = test.shape[0:2]
    all_pred = np.zeros((n_trial, n_window, 5))
    for i in range(n_trial):
        all_pred[i, :, :] = model.predict(test[i, :, :, :])
    return all_pred


def extract_latent(spike_data, lfp_data, intermediate_layer):
    """
    Extract latent representation of decoding model.

    :param spike_data: (3d numpy array) spike train data of format [trial, neuron, time]
    :param lfp_data: (3d numpy array ) LFP data of format [trial, tetrode, time]
    :param intermediate_layer: (keras) function that outputs last hidden layer
    :return: all_latent: (3d numpy array) latent space of format [trial, window, dim]
    """
    spike_stack = stack_data(spike_data, 25, 10)
    lfp_stack = stack_data(lfp_data, 25, 10)
    n_trial, n_window = spike_stack.shape[:2]
    all_latent = np.zeros((n_trial, n_window, 10))
    for i in range(n_window):
        test_data = organize_tetrode(spike_stack[:, i, :, :], lfp_stack[:, i, :, :])
        latent = intermediate_layer.predict(test_data)
        all_latent[:, i, :] = latent
    return all_latent
