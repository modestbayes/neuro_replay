# Rat hippocampus decoding and sequential memory replay

Neuron spike train and local field potential data decoding during experiment: http://www.jneurosci.org/content/36/5/1547

Previous Bayesian statistical model: https://www.ncbi.nlm.nih.gov/pubmed/28529731

Spike train data analysis: https://www.sciencedirect.com/science/article/pii/S0896627316302501

Convolutional neural networks: https://keras.io/getting-started/sequential-model-guide/#sequence-classification-with-1d-convolutions


## neural_decoding

Tetrode-wise convolution: for each tetrode, *convolution filters* are applied on the multivariate time series of spike train and LFP data; *pooled features* from all the tetrodes are combined before a hidden layer and output layer.


<img src="https://raw.githubusercontent.com/modestbayes/neuro_replay/master/tetrode_conv.png" width="500">

`data_utils.py`

`tetrode.py`

`helper.py`


<img src="https://raw.githubusercontent.com/modestbayes/neuro_replay/master/odor_b_aggregate.png" width="800">
