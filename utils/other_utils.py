import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch


def compute_class_weights(y, wt_type='balanced'):
    assert wt_type in ['balanced', 'balanced-sqrt'], 'Weight type not supported'

    pos_count = np.sum(y, axis=0)
    neg_count = y.shape[0] - np.sum(y, axis=0)
    pos_weights = neg_count / pos_count
    if wt_type == 'balanced-sqrt':
        pos_weights = np.sqrt(pos_weights)

    return pos_weights.astype(np.float32)


def compute_metrics(outputs, targets):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs[:, i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[
            i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'acc': compute_accuracy(outputs, targets)}

    return metrics


def compute_accuracy(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size, numClasses]
    :param output: Predicted scores (logits) by the model.
            It should have the same dimensions as target
    :return: accuracy: average accuracy over the samples of the current batch for each condition
    """
    num_samples = target.size(0)
    sigmoid_output = torch.sigmoid(output)
    correct_pred = target.eq(sigmoid_output.round().long())
    accuracy = torch.sum(correct_pred, dim=0)
    return accuracy.cpu().numpy() * (100. / num_samples)


