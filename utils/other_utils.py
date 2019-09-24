import numpy as np
from sklearn.utils import compute_class_weight as sk_compute_class_weight


def compute_class_weights(y, wt_type='balanced'):

    assert wt_type in ['balanced', 'balanced-sqrt'], 'Weight type not supported'

    pos_count = np.sum(y, axis=0)
    neg_count = y.shape[0] - np.sum(y, axis=0)
    pos_weights = neg_count / pos_count
    if wt_type == 'balanced-sqrt':
        pos_weights = np.sqrt(pos_weights)

    return pos_weights
