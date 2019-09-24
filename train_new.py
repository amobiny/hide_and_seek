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


conditions = ['No Find', 'Enlgd Card.', 'Crdmgly', 'Opcty', 'Lsn', 'Edma', 'Cnsldton',
              'Pnumn', 'Atlctss', 'Pnmthrx', 'Plu. Eff.', 'Plu. Othr', 'Frctr', 'S. Dev.']
target_conditions = [3, 13]


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def accuracy_generator(output, target):
    """
     Calculates the classification accuracy.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numClasses]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: accuracy: average accuracy over the samples of the current batch for each condition
    :return: avg_accuracy: average accuracy over all conditions
    """
    batch_size = target.size(0)
    correct_pred = target.eq(output.round().long())
    accuracy = torch.sum(correct_pred, dim=0)
    return accuracy.cpu().numpy() * (100. / batch_size)


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
    epoch_acc = np.zeros((3, len(target_conditions)), dtype='float')  # for Raw/Crop/Drop Images

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
        sum_l2_loss = torch.Tensor([0.]).to(torch.device("cuda"))
        for ii in range(X.size(0)):
            if torch.sum(y[ii]) != 0:
                ck = feature_center[torch.nonzero(y[ii]).view(-1).cpu().numpy()]   # [num_conds, num_att, 768]
                fk = feature_matrix[ii:ii + 1].repeat(ck.shape[0], 1, 1)
                sum_l2_loss += l2_loss(ck, fk)
        batch_loss = loss(y_pred, y.float()) + sum_l2_loss
        epoch_loss[0] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Update Feature Center
        for ii in range(X.size(0)):
            if torch.sum(y[ii]) != 0:
                conds = torch.nonzero(y[ii]).view(-1).cpu().numpy()
                feature_center[conds] += beta * (feature_matrix.detach()[ii:ii+1].repeat(conds.shape[0], 1, 1)
                                                 - feature_center[conds])   # [num_conds, num_att, 768]
        # feature_center[y] += beta * (feature_matrix.detach() - feature_center[y])

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[0] += accuracy_generator(y_pred, y)

        ##################################
        # Attention Cropping
        ##################################
        empty_map_count = 0
        one_nonzero_count = 0
        width_count = 0
        height_count = 0
        with torch.no_grad():
            crop_mask = F.interpolate(attention_map, size=(X.size(2), X.size(3)), mode='bilinear',
                                      align_corners=True) > theta_c
            crop_images = torch.Tensor(X.size(0), attention_map.size(1), X.size(1), crop_size[0], crop_size[1])
            for batch_index in range(crop_mask.size(0)):
                for map_index in range(crop_mask.size(1)):
                    if torch.sum(crop_mask[batch_index, map_index]) == 0:
                        height_min, width_min = 0, 0
                        height_max, width_max = options.input_size, options.input_size
                        # print('0, batch: {}, map: {}'.format(batch_index, map_index))
                        empty_map_count += 1
                    else:
                        nonzero_indices = torch.nonzero(crop_mask[batch_index, map_index, ...])
                        if nonzero_indices.size(0) == 1:
                            height_min, width_min = 0, 0
                            height_max, width_max = options.input_size, options.input_size
                            # print('1, batch: {}, map: {}'.format(batch_index, map_index))
                            one_nonzero_count += 1
                        else:
                            height_min = nonzero_indices[:, 0].min()
                            height_max = nonzero_indices[:, 0].max()
                            width_min = nonzero_indices[:, 1].min()
                            width_max = nonzero_indices[:, 1].max()
                        if width_min == width_max:
                            if width_min == 0:
                                width_max += 1
                            else:
                                width_min -= 1
                            width_count += 1
                        if height_min == height_max:
                            if height_min == 0:
                                height_max += 1
                            else:
                                height_min -= 1
                            height_count += 1
                    crop_images[batch_index, map_index] = F.upsample_bilinear(
                        X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                        size=crop_size)
            crop_images_reshaped = crop_images.view(-1, 3, crop_size[0], crop_size[1])
            print('Batch {} :  empty map: {},  one nonzero idx: {}, width_issue: {}, height_issue: {}'
                  .format(i, empty_map_count, one_nonzero_count, width_count, height_count))
        # crop images forward
        y_pred, _, _ = net(crop_images_reshaped)
        y_pred = y_pred.view(X.shape[0], crop_images.shape[1], num_classes)
        y_pred = torch.mean(y_pred, dim=1)

        # loss
        batch_loss = loss(y_pred, y.float())
        epoch_loss[1] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[1] += accuracy_generator(y_pred, y)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_mask = F.interpolate(attention_map, size=(X.size(2), X.size(3)), mode='bilinear',
                                      align_corners=True) <= theta_d
            drop_images = drop_mask.unsqueeze(2).float() * X.unsqueeze(1)
            drop_images_reshaped = drop_images.view(-1, 3, X.size(2), X.size(3))

        # drop images forward
        y_pred, _, _ = net(drop_images_reshaped)
        y_pred = y_pred.view(X.shape[0], drop_images.shape[1], num_classes)
        y_pred = torch.mean(y_pred, dim=1)

        # loss
        batch_loss = loss(y_pred, y.float())
        epoch_loss[2] += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[2] += accuracy_generator(y_pred, y)

        # end of this batch
        batches += 1
        batch_end = time.time()

        if (i + 1) % verbose == 0:
            raw_acc_str, crop_acc_str, drop_acc_str = '', '', ''
            for ii in range(epoch_acc.shape[1]):
                raw_acc_str += conditions[target_conditions[ii]] + ': ' + str(round(epoch_acc[0, ii] / batches, 2))+',\t'
                crop_acc_str += conditions[target_conditions[ii]] + ': ' + str(round(epoch_acc[1, ii] / batches, 2))+',\t'
                drop_acc_str += conditions[target_conditions[ii]] + ': ' + str(round(epoch_acc[2, ii] / batches, 2))+',\t'

            log_string('\tBatch %d, Time %3.2f' % (i + 1, batch_end - batch_start))
            log_string('\t(Raw) Loss {0:.4f}, Accuracy: '.format(epoch_loss[0]/batches) + raw_acc_str)
            log_string('\t(Crop) Loss {0:.4f}, Accuracy: '.format(epoch_loss[1]/batches) + crop_acc_str)
            log_string('\t(Drop) Loss {0:.4f}, Accuracy: '.format(epoch_loss[2]/batches) + drop_acc_str)

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
    log_string('--' * 25)
    log_string('Running Validation ... ')
    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    raw_loss, crop_loss, epoch_loss = 0, 0, 0
    raw_acc = np.array([0] * len(target_conditions), dtype='float')
    crop_acc = np.array([0] * len(target_conditions), dtype='float')
    epoch_acc = np.array([0] * len(target_conditions), dtype='float')

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
            raw_loss += loss(y_pred_raw, y.float()).item()
            crop_loss += loss(y_pred_crop, y.float()).item()
            epoch_loss += loss(y_pred, y.float()).item()

            # metrics: top-1, top-3, top-5 error
            raw_acc += accuracy_generator(y_pred_raw, y)
            crop_acc += accuracy_generator(y_pred_crop, y)
            epoch_acc += accuracy_generator(y_pred, y)

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
    # Initialize model
    ##################################
    image_size = (options.input_size, options.input_size)
    num_classes = len(target_conditions)
    num_attentions = options.num_attentions
    start_epoch = 0

    feature_net = inception_v3(pretrained=True)
    net = WSDAN_v2(num_classes=num_classes, M=num_attentions, K=options.K, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda"))

    if options.load_model:
        ckpt = options.load_model_path

        if options.initial_training == 0:
            # Get Name (epoch)
            epoch_name = (ckpt.split('/')[-1]).split('.')[0]
            start_epoch = int(epoch_name)

        # Load ckpt and get state_dict
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']

        # Load weights
        net.load_state_dict(state_dict)
        log_string('Network loaded from {}'.format(options.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
            log_string('feature_center loaded from {}'.format(options.ckpt))

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
    os.system('cp {}/models/my_wsdan.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train_new.py {}'.format(BASE_DIR, save_dir))
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
    loss = nn.BCEWithLogitsLoss()

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
