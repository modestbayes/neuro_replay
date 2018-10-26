import numpy as np


def trial_start_end_times(trial_timebins):
    """
    Get the start and end times of trials.

    :param trial_timebins: (numpy array) trial_timebins = data_odor['trialTimeBins'][0, select]
    :return: (2d numpy array) [n_trial, 2] start and end times in each row
    """
    trial_times = []
    n_trial = trial_timebins.shape[0]
    for i in range(n_trial):
        actual_times = trial_timebins[i][:, 0]
        start = actual_times[0]
        end = actual_times[-1]
        trial_times.append([start, end])
    trial_times = np.asarray(trial_times)
    return trial_times


def find_swr_event(swr_start, swr_end, trial_times):
    """
    Helper function for finding SWR events within trials.

    :param swr_start: (float) SWR event start time
    :param swr_end:  (float) SWR event end time
    :param trial_times: (2d numpy array) [n_trial, 2] start and end times in each row
    :return: (int) index of trial during which the SWR event occurs
    """
    n_trials = trial_times.shape[0]
    for i in range(n_trials):
        trial_start, trial_end = trial_times[i, :]
        if swr_start > trial_start and swr_end < trial_end:
            return i
    return -1


def find_all_events(swr_times, trial_times)
    """
    Find all SWR events.
    
    :param swr_times: (numpy array) swr_times = swr_event['swrIndices']
    :param trial_times: (2d numpy array) [n_trial, 2] start and end times in each row
    :return: (1d numpy array) corresponding trial indices
    """
    n_swr = swr_times.shape[0]
    swr_trial = np.zeros(n_swr)
    for i in range(n_swr):
        swr_start, swr_end = swr_times[i, :]
        swr_trial[i] = find_swr_event(swr_start, swr_end, trial_times)
    return swr_trial


def align_swr_times(swr_start, swr_end, trial_start, trial_end):
    """
    Helper function to align SWR times to odor release.
    """
    mid = 0.5 * (trial_start + trial_end)
    return swr_start - mid, swr_end - mid


def align_all_events(swr_trial, swr_times, trial_times):
    """
    Align all SWR events.
    """
    swr_times_aligned = []
    for i, t in enumerate(swr_trial):
        if t != -1:
            swr_start, swr_end = swr_times[i, :]
            trial_start, trial_end = trial_times[int(t), :]
            aligned_start, aligned_end = align_swr_times(swr_start, swr_end, trial_start, trial_end)
            swr_times_aligned.append([aligned_start, aligned_end])
    swr_times_aligned = np.asarray(swr_times_aligned)
    return swr_times_aligned


def align_decoding(n_trial, swr_times_aligned, swr_trial_during):
    """
    Align decoding time with SWR events.
    """
    decoding_alignment = np.zeros((n_trial, 2))
    for i in range(n_trial):
        select_times = swr_times_aligned[swr_trial_during == i, :]
        if select_times.shape[0] > 0:
            if np.sum(select_times[:, 0] > 0.5) > 0:
                select_times = select_times[select_times[:, 0] > 0.5, :]
                k = np.argmax(select_times[:, 1] - select_times[:, 0])
                decoding_alignment[i, :] = select_times[k, :]
    return decoding_alignment
