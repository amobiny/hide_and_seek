"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models import *
from config import options
from utils.other_utils import visualize_attention

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

TOP_K = [1]
# TOP_K = [1, 3, 5]


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100. / batch_size).cpu().numpy())

    return np.array(res)


def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']
    log_string('--' * 25)
    log_string('Running Validation ... ')
    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    raw_loss, crop_loss, epoch_loss = 0, 0, 0
    raw_acc = np.array([0] * len(TOP_K), dtype='float')  # top - 1, 3, 5
    crop_acc = np.array([0] * len(TOP_K), dtype='float')  # top - 1, 3, 5
    epoch_acc = np.array([0] * len(TOP_K), dtype='float')  # top - 1, 3, 5

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            batch_start = time.time()

            # obtain data
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, feature_matrix, attention_map = net(X)

            ##################################
            # Object Localization and Refinement
            ##################################
            upsampled_attention_map = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3)))
            crop_mask = upsampled_attention_map > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()

                # visualize the attention map
                box_coords = [height_min.cpu().numpy(), height_max.cpu().numpy(),
                              width_min.cpu().numpy(), width_max.cpu().numpy()]
                visualize_attention(X[batch_index].clone().cpu().numpy(),
                                    box_coords, upsampled_attention_map[batch_index].clone().cpu().numpy(),
                                    img_save_path=os.path.join(viz_dir, str(i) + '_' + str(batch_index) + '.png'))

                crop_images.append(
                    F.upsample_bilinear(X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                        size=crop_size))
            crop_images = torch.cat(crop_images, dim=0)

            y_pred_crop, _, _ = net(crop_images)

            # final prediction
            y_pred = (y_pred_raw + y_pred_crop) / 2


            # loss
            raw_loss += loss(y_pred_raw, y).item()
            crop_loss += loss(y_pred_crop, y).item()
            epoch_loss += loss(y_pred, y).item()

            # metrics: top-1, top-3, top-5 error
            raw_acc += accuracy(y_pred_raw, y, topk=TOP_K)
            crop_acc += accuracy(y_pred_crop, y, topk=TOP_K)
            epoch_acc += accuracy(y_pred, y, topk=TOP_K)

            # end of this batch
            batches += 1
            batch_end = time.time()
            # if (i + 1) % verbose == 0:
            #     if len(TOP_K) > 1:
            #         log_string('\tBatch %d: Loss %.5f, Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
            #                    (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / batches,
            #                     epoch_acc[2] / batches, batch_end - batch_start))
            #     else:
            #         log_string('\tBatch %d: Loss %.5f, Accuracy: Top-1 %.2f, Time %3.2f' %
            #                    (i + 1, epoch_loss / batches, epoch_acc[0] / batches, batch_end - batch_start))
    # end of validation
    end_time = time.time()

    # metrics for average
    raw_loss /= batches
    raw_acc /= batches
    crop_loss /= batches
    crop_acc /= batches
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    log_string('\tTime %3.2f' % (end_time - start_time))
    if len(TOP_K) > 1:
        log_string('\tLoss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                   (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))
    else:
        log_string('\t LOSS: raw: %.5f, crop: %.5f, combined: %.5f' %
                   (raw_loss, crop_loss, epoch_loss))
        log_string('\t ACCURACY: raw: %.2f, crop: %.2f, combined: %.2f' %
                   (raw_acc[0], crop_acc[0], epoch_acc[0]))

    log_string('--' * 25)
    log_string('')

    return epoch_loss


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    if options.data_name == 'CUB':
        from dataset.dataset_CUB import CUB as data

        data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/NTS_network/CUB_200_2011'
    elif options.data_name == 'cheXpert':
        from dataset.chexpert_dataset import CheXpertDataSet as data

        data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/CheXpert-v1.0-small'
    else:
        raise NameError('Dataset not available!')

    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    viz_dir = os.path.join(save_dir, 'imgs')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    ##################################
    # Initialize model
    ##################################
    image_size = (options.input_size, options.input_size)
    num_classes = options.num_classes
    num_attentions = options.num_attentions

    feature_net = inception_v3(pretrained=True)
    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda"))

    ckpt = options.load_model_path

    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    log_string('Network loaded from {}'.format(ckpt))

    # load feature center
    if 'feature_center' in checkpoint:
        feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
        log_string('feature_center loaded from {}'.format(ckpt))


    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.to(torch.device("cuda"))
    net = nn.DataParallel(net)

    ##################################
    # Load dataset
    ##################################

    train_dataset = data(root=data_dir, is_train=True,
                         input_size=image_size, data_len=options.data_len)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, pin_memory=True,
                              shuffle=True, num_workers=options.workers, drop_last=False)

    test_dataset = data(root=data_dir, is_train=False,
                        input_size=image_size, data_len=options.data_len)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size, pin_memory=True,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00001)
    loss = nn.CrossEntropyLoss()

    ##################################
    # TRAINING
    ##################################

    log_string('')
    log_string('Start Evaluating: Batch size: {}, Test size: {}'.
               format(options.batch_size, len(test_dataset)))

    val_loss = validate(data_loader=test_loader,
                        net=net,
                        loss=loss,
                        verbose=options.verbose)
