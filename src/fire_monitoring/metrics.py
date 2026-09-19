from __future__ import annotations

import numpy as np


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def iou(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    tp = np.logical_and(y_true == cls, y_pred == cls).sum()
    fp = np.logical_and(y_true != cls, y_pred == cls).sum()
    fn = np.logical_and(y_true == cls, y_pred != cls).sum()
    return 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
