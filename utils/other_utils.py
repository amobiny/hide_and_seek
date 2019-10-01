import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


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


def visualize_attention(img, bb_coord, att_map, title='', scan_name='', img_save_path=None):

    # prepare the bounding box
    bb_coord = bb_coord.cpu().numpy()
    width = bb_coord[3] - bb_coord[2]
    height = bb_coord[1] - bb_coord[0]
    rect = patches.Rectangle((bb_coord[2], bb_coord[0]), width, height,
                             linewidth=1.5, edgecolor='r', facecolor='none')

    # prepare the heat-map
    att_map = np.minimum(np.maximum(att_map, 0), 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * np.squeeze(att_map)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # plot the figures
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes[0]
    ax.imshow(np.transpose(img, [1, 2, 0]), cmap='gray')
    ax.imshow(heatmap, alpha=0.4)
    ax.axis('off')
    ax.set_title(title)
    ax.set_xlabel(scan_name)

    ax = axes[1]
    ax.imshow(np.transpose(img, [1, 2, 0]), cmap='gray')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(img_save_path)


def generate_title(y, y_pred, conditions):
    exist_conds = conditions[torch.nonzero(y).cpu().numpy()].reshape(-1)
    pred_conds = conditions[torch.nonzero(torch.sigmoid(y_pred).round()).cpu().numpy()].reshape(-1)
    title = 'Label: '
    if exist_conds.size >= 1:
        for i, cond_name in enumerate(exist_conds):
            title = title + exist_conds[i] + ', '
    else:
        title += '-'
    title += '\n Pred: '
    if pred_conds.size >= 1:
        for i, cond_name in enumerate(pred_conds):
            title = title + pred_conds[i] + ', '
    else:
        title += '-'
    return title

    



