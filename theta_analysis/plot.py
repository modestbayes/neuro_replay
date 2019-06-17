import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from decoding import *


def odor_B_decoding(odor_B_preds, name, use_median=False):
    odor_B_prev = odor_B_preds[:, :, 0]
    odor_B_current = odor_B_preds[:, :, 1]
    odor_B_future = odor_B_preds[:, :, 2]
    
    odor_B_prev_smooth = odor_B_prev.copy()
    odor_B_current_smooth = odor_B_current.copy()
    odor_B_future_smooth = odor_B_future.copy()
    for i in range(odor_B_prev.shape[0]):
        odor_B_prev_smooth[i, :] = gaussian_filter1d(odor_B_prev[i, :], 1)
        odor_B_current_smooth[i, :] = gaussian_filter1d(odor_B_current[i, :], 1)
        odor_B_future_smooth[i, :] = gaussian_filter1d(odor_B_future[i, :], 1)
        
    odor_B_prev_central = central_curves(odor_B_prev_smooth, use_median)
    odor_B_current_central = central_curves(odor_B_current_smooth, use_median)
    odor_B_future_central = central_curves(odor_B_future_smooth, use_median)
    
    x = np.linspace(0, 360, 37)
    plt.plot(x, odor_B_prev_central[0, :], color='deepskyblue', alpha=1, label='P(A)')
    plt.plot(x, odor_B_prev_central[1, :], '--', color='deepskyblue', alpha=0.5)
    plt.plot(x, odor_B_prev_central[2, :], '--', color='deepskyblue', alpha=0.5)

    plt.plot(x, odor_B_current_central[0, :], color='tan', alpha=1, label='P(B)')
    plt.plot(x, odor_B_current_central[1, :], '--', color='tan', alpha=0.5)
    plt.plot(x, odor_B_current_central[2, :], '--', color='tan', alpha=0.5)

    plt.plot(x, odor_B_future_central[0, :], color='mediumseagreen', alpha=1, label='P(C)')
    plt.plot(x, odor_B_future_central[1, :], '--', color='mediumseagreen', alpha=0.5)
    plt.plot(x, odor_B_future_central[2, :], '--', color='mediumseagreen', alpha=0.5)
    #plt.plot([120, 120], [0, 1], '--', color='gray')
    #plt.plot([240, 240], [0, 1], '--', color='gray')
    #plt.ylim(0.1, 0.6)
    plt.xlim(0, 360)
    plt.ylabel('Decoded Probability')
    plt.xlabel('Theta Phase')
    plt.title('{} Odor B'.format(name))
    plt.legend()


def odor_C_decoding(odor_C_preds, name, use_median=False):
    odor_C_prev = odor_C_preds[:, :, 1]
    odor_C_current = odor_C_preds[:, :, 2]
    odor_C_future = odor_C_preds[:, :, 3]
    
    odor_C_prev_smooth = odor_C_prev.copy()
    odor_C_current_smooth = odor_C_current.copy()
    odor_C_future_smooth = odor_C_future.copy()
    for i in range(odor_C_prev.shape[0]):
        odor_C_prev_smooth[i, :] = gaussian_filter1d(odor_C_prev[i, :], 1)
        odor_C_current_smooth[i, :] = gaussian_filter1d(odor_C_current[i, :], 1)
        odor_C_future_smooth[i, :] = gaussian_filter1d(odor_C_future[i, :], 1)
        
    odor_C_prev_central = central_curves(odor_C_prev_smooth, use_median)
    odor_C_current_central = central_curves(odor_C_current_smooth, use_median)
    odor_C_future_central = central_curves(odor_C_future_smooth, use_median)
    
    x = np.linspace(0, 360, 37)
    plt.plot(x, odor_C_prev_central[0, :], color='tan', alpha=1, label='P(B)')
    plt.plot(x, odor_C_prev_central[1, :], '--', color='tan', alpha=0.5)
    plt.plot(x, odor_C_prev_central[2, :], '--', color='tan', alpha=0.5)

    plt.plot(x, odor_C_current_central[0, :], color='mediumseagreen', alpha=1, label='P(C)')
    plt.plot(x, odor_C_current_central[1, :], '--', color='mediumseagreen', alpha=0.5)
    plt.plot(x, odor_C_current_central[2, :], '--', color='mediumseagreen', alpha=0.5)

    plt.plot(x, odor_C_future_central[0, :], color='purple', alpha=1, label='P(D)')
    plt.plot(x, odor_C_future_central[1, :], '--', color='purple', alpha=0.5)
    plt.plot(x, odor_C_future_central[2, :], '--', color='purple', alpha=0.5)
    #plt.plot(x, np.cos(x * np.pi / 180.0) * 0.05 + 0.1, '--', color='gray', label='LFP')
    #plt.plot([120, 120], [0, 1], color='black')
    #plt.plot([240, 240], [0, 1], color='black')
    #plt.ylim(0.1, 0.65)
    plt.xlim(0, 360)
    plt.ylabel('Decoded Probability')
    plt.xlabel('Theta Phase')
    plt.title('{} Odor C'.format(name))
    plt.legend()


def rolling_decoding(odor_B_preds, odor_C_preds, name, use_median=False):
    odor_B_prev = odor_B_preds[:, :, 0]
    odor_B_current = odor_B_preds[:, :, 1]
    odor_B_future = odor_B_preds[:, :, 2]
    
    odor_B_prev_smooth = odor_B_prev.copy()
    odor_B_current_smooth = odor_B_current.copy()
    odor_B_future_smooth = odor_B_future.copy()
    for i in range(odor_B_prev.shape[0]):
        odor_B_prev_smooth[i, :] = gaussian_filter1d(odor_B_prev[i, :], 1)
        odor_B_current_smooth[i, :] = gaussian_filter1d(odor_B_current[i, :], 1)
        odor_B_future_smooth[i, :] = gaussian_filter1d(odor_B_future[i, :], 1)
        
    odor_C_prev = odor_C_preds[:, :, 1]
    odor_C_current = odor_C_preds[:, :, 2]
    odor_C_future = odor_C_preds[:, :, 3]
    
    odor_C_prev_smooth = odor_C_prev.copy()
    odor_C_current_smooth = odor_C_current.copy()
    odor_C_future_smooth = odor_C_future.copy()
    for i in range(odor_C_prev.shape[0]):
        odor_C_prev_smooth[i, :] = gaussian_filter1d(odor_C_prev[i, :], 1)
        odor_C_current_smooth[i, :] = gaussian_filter1d(odor_C_current[i, :], 1)
        odor_C_future_smooth[i, :] = gaussian_filter1d(odor_C_future[i, :], 1)
        
    odor_prev = np.concatenate([odor_B_prev_smooth, odor_C_prev_smooth])
    odor_current = np.concatenate([odor_B_current_smooth, odor_C_current_smooth])
    odor_future = np.concatenate([odor_B_future_smooth, odor_C_future_smooth])
    
    odor_prev_central = central_curves(odor_prev, use_median)
    odor_current_central = central_curves(odor_current, use_median)
    odor_future_central = central_curves(odor_future, use_median)
    
#    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    x = np.linspace(0, 360, 37)
    #plt.plot(x, np.cos(x * np.pi / 180.0) * 0.15 + 0.7, color='gray')
    plt.plot(x, odor_prev_central[0, :], color='red', alpha=1, label='Past')
    plt.plot(x, odor_prev_central[1, :], '--', color='red', alpha=0.5)
    plt.plot(x, odor_prev_central[2, :], '--', color='red', alpha=0.5)
    plt.plot(x, odor_current_central[0, :], color='green', alpha=1, label='Current')
    plt.plot(x, odor_current_central[1, :], '--', color='green', alpha=0.5)
    plt.plot(x, odor_current_central[2, :], '--', color='green', alpha=0.5)
    plt.plot(x, odor_future_central[0, :], color='blue', alpha=1, label='Future')
    plt.plot(x, odor_future_central[1, :], '--', color='blue', alpha=0.5)
    plt.plot(x, odor_future_central[2, :], '--', color='blue', alpha=0.5)
    #plt.plot([120, 120], [0, 1], '--', color='gray')
    #plt.plot([240, 240], [0, 1], '--', color='gray')
    #plt.ylim(0, 0.6)
    plt.xlim(0, 360)
    plt.ylabel('Decoded Probability')
    plt.xlabel('Theta Phase')
    plt.legend()
    plt.title(name)
#    plt.show()
