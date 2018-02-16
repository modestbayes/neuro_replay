import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def stack_test_data(all_data, window_size, stride):
    """Divide spike train data into smaller time windows with stride and stack them for testing.
        
    Args:
        all_data: 3d numpy array of format [trial, neuron, timebin]
        window_size: (int) width of time window
        stride: (int) as stride in convolution
        
    Returns:
        test_data: 4d numpy array of format [trial, time, neuron, window_size]
    """
    n_trial, n_neuron, n_timebin = all_data.shape
    n_example = (n_timebin - window_size) / stride
    test_data = np.zeros((n_trial, n_example, n_neuron, window_size))
    for i in range(n_example):
        window_start = i * stride
        window_end = i * stride + window_size
        test_data[:, i, :, :] = all_data[:, :, window_start:window_end]
    return(test_data)


def predict_trial(model, trial_data):
    """Predict odor probabilities for one trial.
        
    Args:
        model: keras model
        trial_data: 3d numpy array of format [time, neuron, window_size]
        
    Returns:
        predictions: 2d numpy array of format [time, odor]
    """
    trial_data = np.expand_dims(trial_data, axis=1)
    predictions = model.predict(trial_data)
    return(predictions)


def predict_all(model, all_data, window_size, stride):
    """Predict odor probabilities for all trials.
    
    Args:
        all_data: 3d numpy array of format [trial, neuron, timebin]
        window_size: (int) width of time window
        stride: (int) as stride in convolution
        
    Returns:
        predictions: 3d numpy array of format [trial, time, odor]
    """
    test = stack_test_data(all_data, window_size, stride)
    n_trial, n_example = test.shape[0:2]
    all_pred = np.zeros((n_trial, n_example, 5))
    for i in range(n_trial):
        all_pred[i, :, :] = predict_trial(model, test[i, :, :, :])
    return(all_pred)


def plot_trial_pred(predictions):
    """Plot predictions over time for one trial.
    
    Args:
        predictions: 2d numpy array of format [example, odor]
    """
    fig = plt.figure(figsize=(15, 2), dpi=200)
    for j in range(5):
        plt.subplot(int('15' + str(j + 1)))
        plt.ylim(0, 1)
        plt.plot(predictions[:, j])
    plt.show()
    
    
def plot_by_odor(odor_index, all_pred, target):
    """Plot predictions for all trials of the same odor.
    
    Args:
        odor_index: (int) odor by which to group
        all_pred: 3d numpy array of format [trial, time, odor]
        target: 2d numpy array of format [trial, odor]
    """
    odor_names = ['A', 'B', 'C', 'D', 'E']
    pred_by_odor = all_pred[(target[:, odor_index] == 1), :, :]
    n_trial = pred_by_odor.shape[0]
    fig = plt.figure(figsize=(20, 3), dpi=200)
    for j in range(5):
        ax = plt.subplot(int('15' + str(j + 1)))
        plt.xticks([9.5, 19.5, 29.5], ['-1s', '0s', '1s'])
        plt.ylim(0, 1)
        rect1 = patches.Rectangle((21, 0), 2.5, 1, linewidth=1, edgecolor='none', facecolor='red', alpha=0.2)
        ax.add_patch(rect1)
        ax.text(5, 0.75, r'$P({})$'.format(odor_names[j]), fontsize=14)
        for i in range(n_trial):
            plt.plot(pred_by_odor[i, :, j], color='blue', alpha=0.1)
    plt.suptitle('True odor ' + odor_names[odor_index], fontsize=14, fontweight='bold')
    plt.show()