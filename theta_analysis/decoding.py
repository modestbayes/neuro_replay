import numpy as np
import statsmodels.api as sm
import statsmodels as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from data import *
from theta import *


def analysis_flow(rat_name, theta_thres=20, corr_thres=0.9, verbose=False):
    """
    Perform analysis workflow for a rat and exclude trials during which theta amplitude is too low.
    """
    trial_info, spike_data, lfp_data = load_data(rat_name)
    target, spike_data, lfp_data = clean_data(trial_info, spike_data, lfp_data)
    tetrode_index = reference_tetrode(rat_name)
    lfp_reference = lfp_data[:, tetrode_index, :]  # get reference LFP
    
    training_features = process_spike(spike_data, lfp_reference)  # align spike train with LFP
    scaler = StandardScaler()
    training = scaler.fit_transform(training_features)
    
    amp_dist = amplitude_distribution(lfp_reference)
    select_trial = amp_dist > np.percentile(amp_dist, theta_thres)  # select trials by amplitude
    #linear_circular_corr = correlate_time_phase(spike_data, lfp_reference)
    #select_cell = linear_circular_corr > corr_thres  # select cells by correlation
    
    #training_select = training[select_trial][:, select_cell]
    training_select = training[select_trial]
    target_select = target[select_trial]
    # train model
    model = LogisticRegressionCV(cv=5, multi_class='multinomial', 
                                 penalty='l1', solver='saga', 
                                 class_weight='balanced', max_iter=500)
    model = model.fit(training_select, target_select)
    
    if verbose:
        print(np.mean(model.predict(training_select) == target_select))
        
    rolling_preds = np.zeros((target.shape[0], 37, 4))  # rolling prediction during theta cycle
    for i in range(37):  
        rolling_features = process_spike(spike_data, lfp_reference, [300 + i * 10, 420 + i * 10])
        decoding = scaler.transform(rolling_features)
        #decoding_preds = model.predict_proba(decoding[:, select_cell])
        decoding_preds = model.predict_proba(decoding)
        rolling_preds[:, i, :] = decoding_preds
    
    phase_preds = []
    for i in range(3):
        current_features = process_spike(spike_data, lfp_reference, [360 + i * 120, 480 + i * 120])
        decoding = scaler.transform(current_features)
        #current_preds = model.predict_proba(decoding[:, select_cell])
        current_preds = model.predict_proba(decoding)
        phase_preds.append(current_preds[select_trial])
    
    return rolling_preds[select_trial], phase_preds, target_select


def functional_central_curves(curves):
    """
    Find the functional median curve and the central 50% range.
    """
    depth = sm.graphics.functional.banddepth(curves, method='MBD')
    ix_depth = np.argsort(depth)[::-1]
    #res = sm.graphics.fboxplot(curves)
    #ix_depth = res[2]
    ix_median = ix_depth[0]
    median_curve = curves[ix_median, :]
    ix_central = ix_depth[:int(0.5 * ix_depth.shape[0])]
    central_curves = curves[ix_central, :]
    central_min = np.min(central_curves, axis=0)
    central_max = np.max(central_curves, axis=0)
    central_curves = np.stack([median_curve, central_min, central_max])
    return central_curves


def central_curves(curves, use_median=False):
    """
    Calculate usual central curves.
    """
    n = curves.shape[0]
    error = np.std(curves, axis=0) / np.sqrt(n)
    if use_median:
        median_curve = np.percentile(curves, 50, axis=0)
        #central_min = np.percentile(curves, 25, axis=0)
        #central_max = np.percentile(curves, 75, axis=0)
        central_min = median_curve - 2 * error
        central_max = median_curve + 2 * error
        central_curves = np.stack([median_curve, central_min, central_max])
    else:
        mean_curve = np.mean(curves, axis=0)
        central_min = mean_curve - 2 * error
        central_max = mean_curve + 2 * error
        central_curves = np.stack([mean_curve, central_min, central_max])
    return central_curves
