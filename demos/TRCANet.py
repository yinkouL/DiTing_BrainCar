# -*- coding: utf-8 -*-
"""
TRCA-Net SSVEP Classification Demo.

TRCA-Net integrates TRCA (Task-Related Component Analysis) spatial filtering
into a deep neural network. It can be used with or without template signals.

This demo uses the Wang2016 SSVEP benchmark dataset.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices,
    match_kfold_indices)
from metabci.brainda.algorithms.deep_learning.trcanet import TRCANet

# ==================== Dataset & Paradigm ====================
dataset = Wang2016()

paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(0.14, 0.64)],  # delay=0.14s, duration=0.5s
    srate=250
)


def raw_hook(raw, caches):
    # bandpass filter 5-90Hz
    raw.filter(5, 90, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


def epochs_hook(epochs, caches):
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches


def data_hook(X, y, meta, caches):
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches


paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

# ==================== Load Data ====================
X, y, meta = paradigm.get_data(
    dataset,
    subjects=[1],
    return_concat=True,
    n_jobs=None,
    verbose=False)

n_classes = len(dataset.events)  # 40 classes for Wang2016

# ==================== Cross Validation ====================
set_random_seeds(38)
kfold = 6
indices = generate_kfold_indices(meta, kfold=kfold)

# Initialize TRCA-Net estimator
# n_spatial_filters: number of TRCA-like spatial filters (default 8)
# n_time_filters: number of temporal convolution filters (default 16)
estimator = TRCANet(
    X.shape[1],  # n_channels
    X.shape[2],  # n_samples
    n_classes,
    n_spatial_filters=8,
    n_time_filters=16,
    time_kernel=9,
    dropout_rate=0.5,
)

accs = []
for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))

    X_train, y_train = X[train_ind], y[train_ind]
    X_test, y_test = X[test_ind], y[test_ind]

    # Build template T from training data (class-wise average)
    # T shape: (n_batch, n_channels, n_classes, n_samples)
    n_channels, n_samples = X_train.shape[1], X_train.shape[2]
    unique_classes = np.unique(y_train)
    T_template = np.zeros((n_channels, len(unique_classes), n_samples))

    for i, cls in enumerate(unique_classes):
        T_template[:, i, :] = np.mean(X_train[y_train == cls], axis=0)

    # Repeat template for each training sample
    T_train = np.tile(T_template[np.newaxis, ...], (len(X_train), 1, 1, 1))

    # Train with template (use dict format like ConvCA)
    dict_train = {'X': X_train, 'T': T_train}
    estimator.fit(dict_train, y_train)

    # Predict on test data
    T_test = np.tile(T_template[np.newaxis, ...], (len(X_test), 1, 1, 1))
    dict_test = {'X': X_test, 'T': T_test}
    p_labels = estimator.predict(dict_test)

    accs.append(np.mean(p_labels == y_test))

print("TRCA-Net LOO Acc: {:.2f}%".format(np.mean(accs) * 100))
# If everything is fine, you will get the accuracy approximately.
