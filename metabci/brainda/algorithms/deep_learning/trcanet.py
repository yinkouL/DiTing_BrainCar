# -*- coding: utf-8 -*-
#
# Authors: MetaBCI Contributors
# Date: 2024/06/01
# License: MIT License
"""
TRCA-Net: Task-Related Component Analysis Network for SSVEP-based BCI.

TRCA-Net integrates the TRCA (Task-Related Component Analysis) spatial filtering
principle into a deep neural network architecture. It learns spatial filters that
maximize inter-trial similarity, combined with temporal convolutional feature
extraction and correlation-based classification.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from .base import (
    _glorot_weight_zero_bias,
    compute_same_pad2d,
    SkorchNet,
)


class _CorrLayer(nn.Module):
    """Correlation layer for computing normalized cross-correlation between
    signal features and class templates.

    Computes: corr = (X · T) / (||X|| * ||T||)
    """

    def __init__(self):
        super(_CorrLayer, self).__init__()

    def forward(self, X, T):
        # X: (n_batch, n_filters, 1, n_samples)
        # T: (n_batch, n_filters, n_classes, n_samples)
        T = torch.swapaxes(T, -1, -2)  # (n_batch, n_filters, n_samples, n_classes)
        corr_xt = torch.matmul(X, T)  # (n_batch, n_filters, 1, n_classes)
        corr_xx = torch.sum(torch.square(X), -1, keepdim=True)  # (n_batch, n_filters, 1, 1)
        corr_tt = torch.sum(torch.square(T), -2, keepdim=True)  # (n_batch, n_filters, 1, n_classes)
        corr = corr_xt / (torch.sqrt(corr_xx) * torch.sqrt(corr_tt) + 1e-8)
        return corr


@SkorchNet  # TODO: Bug Fix required: unable to make docs with this wrapper
class TRCANet(nn.Module):
    """
    TRCA-Net is a deep learning model for SSVEP-based BCI that combines
    TRCA (Task-Related Component Analysis) spatial filtering principles
    with convolutional neural networks. [1]_

    The architecture consists of:
    1. **Spatial Filter Block**: Learns multiple spatial filters (analogous to
       TRCA components) to project multi-channel EEG into a lower-dimensional
       spatially filtered space.
    2. **Temporal Feature Extraction**: 1D temporal convolutions extract
       time-domain features from the spatially filtered signals.
    3. **Template Processing**: Optional reference/template signal processing
       branch for correlation-based classification.
    4. **Correlation Layer**: Computes normalized cross-correlation between
       input features and class templates.
    5. **Classification Head**: Fully connected layer for final class prediction.

    author: MetaBCI Contributors

    Created on: 2024-06-01

    update log:
        2024-06-01 by MetaBCI Contributors

    Parameters
    ----------
    n_channels: int
        Lead count for the input signal.
    n_samples: int
        Sampling points of the input signal. The value equals
        sampling rate (Hz) * signal duration (s).
    n_classes: int
        The number of classes of input signals to be classified.
    n_spatial_filters: int, optional
        Number of TRCA-like spatial filters to learn. Default is 8.
    n_time_filters: int, optional
        Number of temporal convolution filters. Default is 16.
    time_kernel: int, optional
        Temporal convolution kernel size. Default is 9.
    dropout_rate: float, optional
        Dropout probability. Default is 0.5.

    Attributes
    ----------
    spatial_filter: torch.nn.Sequential
        Spatial filtering block (TRCA-like channel-wise convolution).
    temporal_conv: torch.nn.Sequential
        Temporal feature extraction block.
    template_cnn: torch.nn.Sequential
        Template/reference signal processing block.
    corr_layer: _CorrLayer
        Correlation computation layer.
    fc_layer: torch.nn.Linear
        Final classification layer.

    Examples
    ----------
    >>> # Without template (standard SkorchNet usage):
    >>> num_classes = 2
    >>> estimator = TRCANet(X.shape[1], X.shape[2], num_classes)
    >>> estimator.fit(X[train_index], y[train_index])
    >>>
    >>> # With template (like ConvCA):
    >>> estimator = TRCANet(X.shape[1], X.shape[2], 2)
    >>> dict_ = {'X': train_X, 'T': T}
    >>> estimator.fit(dict_, train_Y)

    See Also
    ----------
    _reset_parameters: Initialize the model parameters

    References
    ----------
    .. [1] Tanaka H, Katura T, Sato H. Task-related component analysis for
       functional neuroimaging and application to near-infrared spectroscopy data.
       NeuroImage, 2013.
    """

    def __init__(
        self,
        n_channels,
        n_samples,
        n_classes,
        n_spatial_filters=8,
        n_time_filters=16,
        time_kernel=9,
        dropout_rate=0.5,
    ):
        super().__init__()

        # Store parameters
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_spatial_filters = n_spatial_filters
        self.n_time_filters = n_time_filters
        self.time_kernel = time_kernel
        self.dropout_rate = dropout_rate

        # Step 1: Spatial Filter Block (TRCA-like)
        # Input:  (batch, 1, n_channels, n_samples)
        # Output: (batch, n_spatial_filters, 1, n_samples)
        # This conv layer learns spatial filters that act like TRCA components,
        # projecting all channels down to a single value per filter.
        self.spatial_filter = nn.Sequential(
            OrderedDict(
                [
                    (
                        "spatial_conv",
                        nn.Conv2d(
                            1,
                            n_spatial_filters,
                            (n_channels, 1),
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(n_spatial_filters)),
                ]
            )
        )

        # Step 2: Temporal Feature Extraction Block
        # Input:  (batch, n_spatial_filters, 1, n_samples)
        # Output: (batch, n_time_filters, 1, n_samples)
        self.temporal_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "same_padding",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (1, n_samples),
                                (1, time_kernel),
                                stride=(1, 1),
                            ),
                            0,
                        ),
                    ),
                    (
                        "time_conv",
                        nn.Conv2d(
                            n_spatial_filters,
                            n_time_filters,
                            (1, time_kernel),
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(n_time_filters)),
                    ("elu", nn.ELU()),
                    ("drop", nn.Dropout(dropout_rate)),
                ]
            )
        )

        # Step 3: Template Processing Block
        # Input:  (batch, n_spatial_filters, n_classes, n_samples)
        # Output: (batch, n_time_filters, n_classes, n_samples)
        self.template_cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "same_padding",
                        nn.ConstantPad2d(
                            compute_same_pad2d(
                                (n_classes, n_samples),
                                (1, time_kernel),
                                stride=(1, 1),
                            ),
                            0,
                        ),
                    ),
                    (
                        "time_conv",
                        nn.Conv2d(
                            n_spatial_filters,
                            n_time_filters,
                            (1, time_kernel),
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(n_time_filters)),
                    ("elu", nn.ELU()),
                    ("drop", nn.Dropout(dropout_rate)),
                ]
            )
        )

        # Step 4: Correlation Layer
        self.corr_layer = _CorrLayer()

        # Step 5: Classification Head
        # Correlation output: (batch, n_time_filters, 1, n_classes)
        # After flatten: (batch, n_time_filters * n_classes)
        self.flatten_layer = nn.Flatten()
        self.fc_layer = nn.Linear(n_time_filters * n_classes, n_classes)

        # Fallback classifier (used when no template T is provided)
        self.fallback_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fallback_fc = nn.Linear(n_time_filters, n_classes)

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        """Initialize model parameters using Glorot uniform initialization
        for weights and zeros for biases. BatchNorm weights are set to 1."""
        _glorot_weight_zero_bias(self)

    def _apply_spatial_filter(self, X):
        """Apply the spatial filter block to input signal X.

        Parameters
        ----------
        X: Tensor
            Input signal of shape (n_batch, 1, n_channels, n_samples).

        Returns
        -------
        Tensor
            Spatially filtered signal of shape (n_batch, n_spatial_filters, 1, n_samples).
        """
        return self.spatial_filter(X)

    def _process_template(self, T, n_batch, n_chan, n_classes, n_samp):
        """Reshape and apply spatial + temporal processing to template signals.

        Parameters
        ----------
        T: Tensor
            Template of shape (n_batch, n_channels, n_classes, n_samples).
        n_batch, n_chan, n_classes, n_samp: int
            Shape dimensions of the template.

        Returns
        -------
        Tensor
            Processed template of shape (n_batch, n_time_filters, n_classes, n_samples).
        """
        # Reshape template to apply spatial filter:
        # (batch, chan, classes, samp) -> (batch*classes, 1, chan, samp)
        T = T.permute(0, 2, 1, 3).contiguous()
        T = T.view(n_batch * n_classes, 1, n_chan, n_samp)
        T = self.spatial_filter(T)  # (batch*classes, n_spatial_filters, 1, samp)
        # Reshape back: -> (batch, n_spatial_filters, classes, samp)
        T = T.view(n_batch, n_classes, self.n_spatial_filters, 1, n_samp)
        T = T.permute(0, 2, 1, 3, 4).contiguous()
        T = T.squeeze(-2)  # (batch, n_spatial_filters, classes, samp)
        # Apply temporal processing
        T = self.template_cnn(T)  # (batch, n_time_filters, classes, samp)
        return T

    def forward(self, X, T=None):
        """Forward pass of TRCA-Net.

        Parameters
        ----------
        X: Tensor
            Input EEG signal of shape (n_batch, n_channels, n_samples).
        T: Tensor, optional
            Template/reference signal of shape (n_batch, n_channels, n_classes, n_samples).
            If provided, correlation-based classification is used.
            If not provided, a pooled feature-based classification is used.

        Returns
        -------
        Tensor
            Classification output of shape (n_batch, n_classes).
        """
        # Add channel dimension for Conv2d: (batch, 1, n_channels, n_samples)
        X = X.unsqueeze(1)

        # Apply spatial filtering: (batch, n_spatial_filters, 1, n_samples)
        X = self.spatial_filter(X)

        if T is not None:
            # Template-based classification path (full TRCA-Net)
            n_batch, n_chan, n_classes, n_samp = T.shape
            T = self._process_template(T, n_batch, n_chan, n_classes, n_samp)
            # T: (batch, n_time_filters, n_classes, n_samples)

            # Apply temporal convolution to signal
            X = self.temporal_conv(X)  # (batch, n_time_filters, 1, n_samples)

            # Compute correlation between signal features and templates
            corr = self.corr_layer(X, T)  # (batch, n_time_filters, 1, n_classes)
            corr = self.flatten_layer(corr)  # (batch, n_time_filters * n_classes)
            out = self.fc_layer(corr)
        else:
            # Fallback: standard classification path (no template)
            # Apply temporal convolution
            X = self.temporal_conv(X)  # (batch, n_time_filters, 1, n_samples)
            # Global average pooling
            X = self.fallback_pool(X)  # (batch, n_time_filters, 1, 1)
            X = X.squeeze(-1).squeeze(-1)  # (batch, n_time_filters)
            out = self.fallback_fc(X)

        return out

    def cal_backbone(self, X, T=None, **kwargs):
        """Extract backbone features before the final classification layer.

        Parameters
        ----------
        X: Tensor
            Input EEG signal of shape (n_batch, n_channels, n_samples).
        T: Tensor, optional
            Template signal of shape (n_batch, n_channels, n_classes, n_samples).

        Returns
        -------
        Tensor
            Feature representation before classification.
        """
        X = X.unsqueeze(1)
        X = self.spatial_filter(X)

        if T is not None:
            n_batch, n_chan, n_classes, n_samp = T.shape
            T = self._process_template(T, n_batch, n_chan, n_classes, n_samp)
            X = self.temporal_conv(X)
            corr = self.corr_layer(X, T)
            backbone = self.flatten_layer(corr)
        else:
            X = self.temporal_conv(X)
            backbone = self.fallback_pool(X)
            backbone = backbone.squeeze(-1).squeeze(-1)

        return backbone
