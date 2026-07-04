import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import time
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices,
    match_kfold_indices)
from metabci.brainda.algorithms.decomposition import DCPM


dataset = Wang2016()

paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 0.64)],
    srate=250
)


# add 5-90Hz bandpass filter in raw hook
def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(5, 90, l_trans_bandwidth=2,h_trans_bandwidth=5,
        phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


def epochs_hook(epochs, caches):
    # do something with epochs object
    # print(epochs.event_id)
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches


def data_hook(X, y, meta, caches):
    # retrive caches from the last stage
    # print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
    # do something with X, y, and meta
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches


paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

X, y, meta = paradigm.get_data(
    dataset,
    subjects=[1],
    return_concat=True,
    n_jobs=None,
    verbose=False)

# 6-fold cross validation
set_random_seeds(38)
kfold = 6
indices = generate_kfold_indices(meta, kfold=kfold)

def online_predict(model, X_test):
    """Simulate online decoding by sending one trial at a time."""
    p_labels = []
    predict_times = []
    for trial in X_test:
        trial = np.copy(trial[np.newaxis, ...])
        start_time = time.perf_counter()
        p_label = model.predict(trial)
        predict_times.append(time.perf_counter() - start_time)
        p_labels.append(p_label[0])
    return np.array(p_labels), np.array(predict_times)


# classifier
estimator = DCPM(n_components=8, online_flag=True)


accs = []
online_times = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    model = estimator.fit(np.copy(X[train_ind]), y[train_ind])
    p_labels, predict_times = online_predict(model, X[test_ind])

    accs.append(np.mean(p_labels==y[test_ind]))
    online_times.extend(predict_times)
    print("Fold {} online acc: {:.4f}, avg predict time: {:.4f} ms/trial".format(
        k + 1, accs[-1], np.mean(predict_times) * 1000))
print("Average online acc: {:.4f}".format(np.mean(accs)))
print("Average online predict time: {:.4f} ms/trial".format(np.mean(online_times) * 1000))


# If everything is fine, you will get the accuracy about 0.9417.
