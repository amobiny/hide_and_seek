"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models import *
from datetime import datetime
from utils.logger_utils import Logger
from config import options

TOP_K = [1]
# TOP_K = [1, 3, 5]

conditions = ['No Find', 'Enlgd Card.', 'Crdmgly', 'Opcty', 'Lsn', 'Edma', 'Cnsldton',
              'Pnumn', 'Atlctss', 'Pnmthrx', 'Plu. Eff.', 'Plu. Othr', 'Frctr', 'S. Dev.']
target_conditions = [13]


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


def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    feature_center = kwargs['feature_center']
    epoch = kwargs['epoch']
    save_freq = kwargs['save_freq']
    model_dir = kwargs['model_dir']
    logs_dir = kwargs['logs_dir']
    verbose = kwargs['verbose']

    # Attention Regularization: LA Loss
    l2_loss = nn.MSELoss()

    # Default Parameters
    beta = 1e-4
    theta_c = 0.5
    theta_d = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = np.array([0, 0, 0], dtype='float')  # Loss on Raw/Crop/Drop Images
    epoch_acc = np.zeros((3, len(TOP_K)), dtype='float')  # Top-1/3/5 Accuracy for Raw/Crop/Drop Images

    # begin training
    start_time = time.time()
    log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()
    for i, (X, y) in enumerate(data_loader):
        batch_start = time.time()

        # obtain data for training
        X = X.to(torch.device("cuda"))
        y = y.to(torch.device("cuda"))

        ##################################
        # Raw Image
        ##################################
        y_pred, feature_matrix, attention_map = net(X)

        # loss
        batch_loss = loss(y_pred, y) + l2_loss(feature_matrix, feature_center[y])
        epoch_loss[0] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Update Feature Center
        feature_center[y] += beta * (feature_matrix.detach() - feature_center[y])

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[0] += accuracy(y_pred, y, topk=TOP_K)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
                crop_images.append(
                    F.upsample_bilinear(X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                        size=crop_size))
            crop_images = torch.cat(crop_images, dim=0)

        # crop images forward
        y_pred, _, _ = net(crop_images)

        # loss
        batch_loss = loss(y_pred, y)
        epoch_loss[1] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[1] += accuracy(y_pred, y, topk=TOP_K)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) <= theta_d
            drop_images = X * drop_mask.float()

        # drop images forward
        y_pred, _, _ = net(drop_images)

        # loss
        batch_loss = loss(y_pred, y)
        epoch_loss[2] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[2] += accuracy(y_pred, y, topk=TOP_K)

        # end of this batch
        batches += 1
        batch_end = time.time()
        if (i + 1) % verbose == 0:
            if len(TOP_K) > 1:
                log_string(
                    '\tBatch %d: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, '
                    'Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f' %
                    (i + 1,
                     epoch_loss[0] / batches, epoch_acc[0, 0] / batches, epoch_acc[0, 1] / batches,
                     epoch_acc[0, 2] / batches,
                     epoch_loss[1] / batches, epoch_acc[1, 0] / batches, epoch_acc[1, 1] / batches,
                     epoch_acc[1, 2] / batches,
                     epoch_loss[2] / batches, epoch_acc[2, 0] / batches, epoch_acc[2, 1] / batches,
                     epoch_acc[2, 2] / batches,
                     batch_end - batch_start))
            else:
                log_string(
                    '\tBatch %d: (Raw) Loss %.4f, Accuracy: %.2f, (Crop) Loss %.4f, '
                    'Accuracy: %.2f, (Drop) Loss %.4f, Accuracy: %.2f, Time %3.2f' %
                    (i + 1,
                     epoch_loss[0] / batches, epoch_acc[0, 0] / batches,
                     epoch_loss[1] / batches, epoch_acc[1, 0] / batches,
                     epoch_loss[2] / batches, epoch_acc[2, 0] / batches,
                     batch_end - batch_start))

    # save checkpoint model
    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': model_dir,
            'state_dict': state_dict,
            'feature_center': feature_center.cpu()},
            os.path.join(model_dir, '%03d.ckpt' % (epoch + 1)))

    # end of this epoch
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    if len(TOP_K) > 1:
        log_string(
            'Train: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, Accuracy: '
            '(%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f' %
            (epoch_loss[0], epoch_acc[0, 0], epoch_acc[0, 1], epoch_acc[0, 2],
             epoch_loss[1], epoch_acc[1, 0], epoch_acc[1, 1], epoch_acc[1, 2],
             epoch_loss[2], epoch_acc[2, 0], epoch_acc[2, 1], epoch_acc[2, 2],
             end_time - start_time))
    else:
        log_string(
            'Train: (Raw) Loss %.4f, Accuracy: %.2f, (Crop) Loss %.4f, Accuracy: '
            '%.2f, (Drop) Loss %.4f, Accuracy: %.2f, Time %3.2f' %
            (epoch_loss[0], epoch_acc[0, 0],
             epoch_loss[1], epoch_acc[1, 0],
             epoch_loss[2], epoch_acc[2, 0],
             end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']
    log_string('--'*25)
    log_string('Running Validation ... ')
    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    raw_loss, crop_loss, epoch_loss = 0, 0, 0
    raw_acc = np.array([0]*len(TOP_K), dtype='float')  # top - 1, 3, 5
    crop_acc = np.array([0]*len(TOP_K), dtype='float')  # top - 1, 3, 5
    epoch_acc = np.array([0]*len(TOP_K), dtype='float')  # top - 1, 3, 5

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):

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
            crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
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

    log_string('--'*25)
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
    # Initialize model
    ##################################
    image_size = (options.input_size, options.input_size)
    num_classes = options.num_classes
    num_attentions = options.num_attentions
    start_epoch = 0

    feature_net = inception_v3(pretrained=True)
    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda"))

    if options.load_model:
        ckpt = options.ckpt

        if options.initial_training == 0:
            # Get Name (epoch)
            epoch_name = (ckpt.split('/')[-1]).split('.')[0]
            start_epoch = int(epoch_name)


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
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of model def
    os.system('cp {}/models/wsdan.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train_wsdan.py {}'.format(BASE_DIR, save_dir))
    if options.data_name == 'cheXpert':
        os.system('cp {}/dataset/chexpert_dataset.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.to(torch.device("cuda"))
    net = nn.DataParallel(net)

    ##################################
    # Load dataset
    ##################################

    train_dataset = data(root=data_dir, is_train=True, target_label=target_conditions,
                         input_size=image_size, data_len=options.data_len)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, pin_memory=True,
                              shuffle=True, num_workers=options.workers, drop_last=False)

    validate_dataset = data(root=data_dir, is_train=False, target_label=target_conditions,
                            input_size=image_size, data_len=options.data_len)
    validate_loader = DataLoader(validate_dataset, batch_size=options.batch_size, pin_memory=True,
                                 shuffle=False, num_workers=options.workers, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00001)
    loss = nn.CrossEntropyLoss()

    ##################################
    # Learning rate scheduling
    ##################################
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # TRAINING
    ##################################
    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(validate_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    val_logger = Logger(os.path.join(logs_dir, 'val'))

    for epoch in range(start_epoch, options.epochs):
        train(epoch=epoch,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              loss=loss,
              optimizer=optimizer,
              save_freq=options.save_freq,
              model_dir=model_dir,
              logs_dir=logs_dir,
              verbose=options.verbose)
        val_loss = validate(data_loader=validate_loader,
                            net=net,
                            loss=loss,
                            verbose=options.verbose)
        scheduler.step()
