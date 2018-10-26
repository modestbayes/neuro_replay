import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from data_utils import *


def predict_all(model, all_data):
    """
    Predict odor probabilities for all trials.

    :param model: (keras) decoding model
    :param all_data: (4d numpy array) data of format [trial, window, neuron, time]
    :return: (3d numpy array) prediction of format [trial, time, odor]
    """
    test = stack_data(all_data, 25, 10)
    n_trial, n_window = test.shape[0:2]
    all_pred = np.zeros((n_trial, n_window, 5))

    for i in range(n_trial):
        all_pred[i, :, :] = model.predict(test[i, :, :, :])

    return all_pred


def extract_latent(intermediate_layer, spike_data, lfp_data, tetrode_ids, tetrode_units, window, stride):
    """
    Extract latent representation of decoding model.

    :param intermediate_layer: (keras) function that outputs last hidden layer
    :param spike_data: (3d numpy array) spike train data of format [trial, neuron, time]
    :param lfp_data: (3d numpy array ) LFP data of format [trial, tetrode, time]
    :param tetrode_ids: (list) of tetrode ids in the order of LFP data
    :param tetrode_units: (dict) number of neuron units on each tetrode
    :param window: (int) time window size must be the same for training the model
    :param stride: (int) moving window stride
    :return: (3d numpy array) latent space of format [trial, window, dim]
    """
    spike_stack = stack_data(spike_data, window, stride)
    lfp_stack = stack_data(lfp_data, window, stride)
    n_trial, n_window = spike_stack.shape[:2]
    all_latent = np.zeros((n_trial, n_window, 10))

    for i in range(n_window):
        test_data = organize_tetrode(spike_stack[:, i, :, :], lfp_stack[:, i, :, :],
                                     tetrode_ids, tetrode_units, verbose=False)
        latent = intermediate_layer.predict(test_data)
        all_latent[:, i, :] = latent

    return all_latent


def latent_models(latent_data, latent_target, decoding_index):
    """
    Create models in latent space: PCA model to reduce dimensionality and logistic regression model for decoding.

    :param latent_data: (3d numpy array) latent space of format [trial, window, dim]
    :param latent_target: (1d numpy array) odor target
    :param decoding_index: (int) which time window to use
    :return: (sklearn) PCA and LR models
    """
    temp = np.split(latent_data, latent_data.shape[1], axis=1)
    latent_stack = np.vstack([x[:, 0, :] for x in temp])

    pca = PCA(n_components=2)
    pca = pca.fit(latent_stack)

    latent_decoding = pca.transform(latent_data[:, decoding_index, :])

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf = clf.fit(latent_decoding, latent_target)

    return pca, clf


def latent_grid(latent_data, pca, clf, h=0.01):
    """
    Create grid in latent for visualization.
    """
    temp = np.split(latent_data, latent_data.shape[1], axis=1)
    latent_stack = np.vstack([x[:, 0, :] for x in temp])

    principal = pca.transform(latent_stack)
    x_max, y_max = np.max(principal, axis=0) + 0.1
    x_min, y_min = np.min(principal, axis=0) - 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    return xx, yy, Z


def align_rolling(trial_idx, time_idx, data_spike, scaler, decoder):
    """
    Predict with a rolling window around SWR events.
    """
    rolling_hat = np.zeros((20, 4))
    for i in range(20):
        current_start = time_idx - 20 + i * 2
        current_end = current_start + 25
        current_window = np.expand_dims(data_spike[trial_idx, :, current_start:current_end], axis=0)
        current_data = scaler.transform(np.mean(current_window, axis=2))
        current_hat = decoder.predict_proba(current_data)
        rolling_hat[i, :] = current_hat
    return rolling_hat
