import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def theta_wave(lfp):
    """
    Run bandpass filter on data to get theta waves.
    """
    return butter_bandpass_filter(lfp, 4, 7, 100)


def amplitude_distribution(lfp_reference):
    """
    Calculate the distribution of local field potential amplitude across trials.
    """
    amp_dist = []
    n = lfp_reference.shape[0]
    for i in range(n):
        theta_mean = theta_wave(lfp_reference[i, :])[200:300]
        analytical_signal = hilbert(theta_mean)
        phase = np.unwrap(np.angle(analytical_signal)) * 180 / np.pi
        amplitude_envelope = np.abs(analytical_signal)
        theta_amp = np.min(amplitude_envelope[(phase > 360) & (phase < 720)])  # minimum amplitude during first theta
        amp_dist.append(theta_amp)
    amp_dist = np.asarray(amp_dist)
    return amp_dist


def correlate_time_phase(spike_data, lfp_reference):
    """
    Calculate linear circular correlation between spikes and theta phases.
    """
    n_trial, n_cell, _ = spike_data.shape
    linear_circular_corr = np.zeros(n_cell)
    for j in range(n_cell):
        spike_time = []
        spike_phase = []
        for i in range(n_trial):  # go through the trials
            theta_mean = theta_wave(lfp_reference[i, :])[200:300]
            analytical_signal = hilbert(theta_mean)
            phase = np.unwrap(np.angle(analytical_signal))
            # count the spikes
            spike_true = spike_data[i, j, 200:300] > 0
            current_time = np.arange(100)[spike_true]
            current_phase = (phase * 180 / np.pi)[spike_true]
            spike_time.append(current_time)
            spike_phase.append(current_phase)
        spike_time = np.concatenate(spike_time)
        spike_phase = np.concatenate(spike_phase)
        reg = LinearRegression(fit_intercept=False).fit(spike_time.reshape(-1, 1), spike_phase)
        score = reg.score(spike_time.reshape(-1, 1), spike_phase)
        linear_circular_corr[j] = np.sqrt(score)
    return linear_circular_corr


def theta_features(spike_data, lfp_reference, trial_index, phase_range):
    """
    Create a feature vector (firing rates for cells) during a theta phase for a specific trial.
    """
    theta_mean = theta_wave(lfp_reference[trial_index, :])[200:300]
    analytical_signal = hilbert(theta_mean)
    phase = np.unwrap(np.angle(analytical_signal))
    phase_degree = phase * 180 / np.pi  # theta phase in degree
    current_spike = spike_data[trial_index, :, 200:300]
    select_spike = current_spike[:, (phase_degree >= phase_range[0]) & (phase_degree <= phase_range[1])]
    if select_spike.shape[1] > 0:
        features = np.mean(select_spike, axis=1)
    else:
        features = np.mean(current_spike, axis=1)  # in case of empty phase
    return features


def process_spike(spike_data, lfp_reference, phase_range=[480, 600]):
    """
    Wrapper for theta_features to process spike train data for all the trials.
    """
    n, d, t = spike_data.shape
    training_features = np.zeros((n, d))
    for i in range(n):
        training_features[i] = theta_features(spike_data, lfp_reference, i, phase_range)
    return training_features
