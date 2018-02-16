import numpy as np
import pandas as pd
import time
from training import build_model, cross_validate_model

# load data
data_X = np.load('/home/linggel/spike_train/spike_train.npy')
train_start_idx = 210
train_num_idx = 25
training = data_X[:, :, train_start_idx:(train_start_idx + train_num_idx)]
training = np.expand_dims(training, axis=1)
training = (training - np.mean(training)) / np.std(training)
target = np.load('/home/linggel/spike_train/odor_target.npy')
print('Rat brain spike train data loaded and processed.')

# set up parameters
latent_range = [5, 10] 
print('Number of filters {}'.format(latent_range))
penalty_range = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
print('Activation penalty {}'.format(penalty_range))
width_range = [10, 25]
print('Pooling width {}'.format(width_range))
dense_range = [0, 5, 10]
print('Dense layer {}'.format(dense_range))
epoch_range = [50, 100, 150, 200, 250]
print('Number of epochs {}'.format(epoch_range))
batch_range = [64]
print('Batch size {}'.format(batch_range))
n_total = len(latent_range) * len(penalty_range) * len(width_range) * len(dense_range) * len(epoch_range) * len(batch_range)
results = np.zeros((n_total, 7))
print('Total number of parameter combinations {}'.format(n_total))

# grid search
current_row = 0
start = time.time()
for l in latent_range:
    for p in penalty_range:
        for w in width_range:
            for d in dense_range:
                model_params = [l, p, w, d]
                for e in epoch_range:
                    for b in batch_range:
                        training_params = [e, b]
                        a = cross_validate_model(training, target, model_params, training_params)
                        results[current_row, :] = np.asarray([l, p, w, d, e, b, a])
                        current_row += 1
                        if current_row % 10 == 0:
                            acc = np.max(results[:, 6])
                            now = time.time()
                            t = now - start
                            print('{x} seconds elapsed {n} models trained...best accuracy {k}'.format(x=t, n=current_row, k=acc))

# results
df = pd.DataFrame(results, columns=['latent', 'penalty', 'pooling', 'dense', 'epochs', 'batch', 'accuracy'])
df.head()
df.to_pickle('/home/linggel/spike_train/results_classic.pkl')