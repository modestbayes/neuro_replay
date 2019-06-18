# Rat hippocampus decoding and sequential memory replay

Neuron spike train and local field potential data decoding during experiment: http://www.jneurosci.org/content/36/5/1547

Previous Bayesian statistical model: https://www.ncbi.nlm.nih.gov/pubmed/28529731

Spike train data analysis: https://www.sciencedirect.com/science/article/pii/S0896627316302501

Convolutional neural networks: https://keras.io/getting-started/sequential-model-guide/#sequence-classification-with-1d-convolutions


## neural_decoding

Tetrode-wise convolution: for each tetrode, *convolution filters* are applied on the multivariate time series of spike train and LFP data; *pooled features* from all the tetrodes are combined before a hidden layer and output layer.

<img src="https://raw.githubusercontent.com/modestbayes/neuro_replay/master/tetrode_conv.png" width="500">

`data_utils.py` process data arrays *(trial_info, spike_data_binned, lfp_data_sampled)* and organize data by tetrode

`tetrode.py` build neural network model in Keras and perform cross-validation during training

`helper.py` extract model hidden layer latent representation within different time windows

**Result visualization**: see notebooks `tetrode_model.ipynb` and `aggregate_decoding.ipynb`

<img src="https://raw.githubusercontent.com/modestbayes/neuro_replay/master/odor_b_aggregate.png" width="800">

## theta_analysis

Theta cycle replay analysis: theta cycles are determined with Hilbert transformation on filtered LFP data; spike train data are then aligned with theta phase rather than time.

`data.py` process data arrays *(trial_info, spike_data_binned, lfp_data_sampled)* and find reference LFP tetrode

`theta.py` calculate theta phase from LFP data and align spike train data

`decoding.py` train LASSO decoding model with cross-validation and get model prediciton

**Final results**: see notebook `final_theta_analysis.ipynb`
