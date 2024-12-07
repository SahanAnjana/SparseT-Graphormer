import datetime
from collections import defaultdict
from datetime import time
import os

import numpy as np
import torch
from src.utils.misc import forecasting_acc, MetricLogger


class PredTestMeter(MetricLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = None
        self.targets = None
        self.target_shape = None

    def store_predictions(self, pred, target):
        """
        used to calculate overall metrics (as a single batch), instead of averaging
        :param pred:
        :param target:
        :return:
        """
        if self.preds is not None and self.targets is not None:
            self.preds = torch.cat(
                tensors=[self.preds, pred],
                dim=0
            )
            self.targets = torch.cat(
                tensors=[self.targets, target],
                dim=0
            )
        else:
            self.preds, self.targets = pred, target

    def finalize_metrics(self, target_shape=None):
        if target_shape:
            target_shape = tuple([self.preds.shape[0]] + list(target_shape[1:]))
        metrics = forecasting_acc(
            self.preds,
            self.targets,
            target_shape
        )
        self.preds = None
        self.targets = None
        return metrics
