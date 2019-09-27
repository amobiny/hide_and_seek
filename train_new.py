"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os

from torchvision.models import densenet121

from utils.other_utils import compute_accuracy, compute_class_weights, compute_metrics

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
from dataset.chexpert_dataset import CheXpertDataSet as data

conditions = ['No Find', 'Enlgd Card.', 'Crdmgly', 'Opcty', 'Lsn', 'Edma', 'Cnsldton',
              'Pnumn', 'Atlctss', 'Pnmthrx', 'Plu. Eff.', 'Plu. Othr', 'Frctr', 'S. Dev.']
# target_conditions = [3, 13]
target_conditions = list(range(14))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train(**kwargs):
    # Retrieve training configuration
    global_step = kwargs['global_step']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    feature_center = kwargs['feature_center']
    model_dir = kwargs['model_dir']
    verbose = kwargs['verbose']
    val_freq = kwargs['val_freq']

    # Attention Regularization: LA Loss
    l2_loss = nn.MSELoss()

    # Default Parameters
    beta = 1e-4
    theta_c = 0.5
    theta_d = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    best_val_loss = 100
    best_val_auc = 0

    for epoch in range(start_epoch, options.epochs):

        # metrics initialization
        batches = 0
        epoch_loss = np.array([0, 0, 0], dtype='float')  # Loss on Raw/Crop/Drop Images
        epoch_acc = np.zeros((3, len(target_conditions)), dtype='float')  # for Raw/Crop/Drop Images

        # begin training
        log_string('**' * 25)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()
        for i, (X, y) in enumerate(data_loader):
            global_step += 1

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
                    ck = feature_center[torch.nonzero(y[ii]).view(-1).cpu().numpy()]  # [num_conds, num_att, 768]
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
                    feature_center[conds] += beta * (feature_matrix.detach()[ii:ii + 1].repeat(conds.shape[0], 1, 1)
                                                     - feature_center[conds])  # [num_conds, num_att, 768]
            # feature_center[y] += beta * (feature_matrix.detach() - feature_center[y])

            with torch.no_grad():
                epoch_acc[0] += compute_accuracy(y_pred, y)

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
                # print('Batch {} :  empty map: {},  one nonzero idx: {}, width_issue: {}, height_issue: {}'
                #       .format(i, empty_map_count, one_nonzero_count, width_count, height_count))
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

            with torch.no_grad():
                epoch_acc[1] += compute_accuracy(y_pred, y)

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

            with torch.no_grad():
                epoch_acc[2] += compute_accuracy(y_pred, y)

            # end of this batch
            batches += 1

            if (i + 1) % verbose == 0:
                log_string('iter {0:<5}: raw_loss={1:.4f}, raw_acc={2:2.2f}, '
                           'crp_loss={3:.4f}, crp_acc={4:2.2f}, '
                           'drp_loss={5:.4f}, drp_acc={6:2.2f}'
                           .format(i + 1, epoch_loss[0] / batches, epoch_acc[0, ii] / batches,
                                   epoch_loss[1] / batches, epoch_acc[1, ii] / batches,
                                   epoch_loss[2] / batches, epoch_acc[2, ii] / batches))

                # write to tensorboard
                info = {'raw_loss': epoch_loss[0] / batches,
                        'crop_loss': epoch_loss[1] / batches,
                        'drop_loss': epoch_loss[2] / batches,
                        'lr': optimizer.param_groups[0]['lr']}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)

            if (i + 1) % val_freq == 0:
                # display results on training data
                log_string('--' * 25)
                str_print = ''
                str_print += '{0:<13} {1:<7} {2:<7} {3:<7}\n'.format('class_name', 'raw', 'crop', 'drop')
                str_print += '--' * 18 + '\n'
                for ii in range(epoch_acc.shape[1]):
                    raw_acc_ii = epoch_acc[0, ii] / batches
                    crop_acc_ii = epoch_acc[1, ii] / batches
                    drop_acc_ii = epoch_acc[2, ii] / batches
                    str_print += '{0:<13} {1:<7.2f} {2:<7.2f} {3:<7.2f}'.format(conditions[ii],
                                                                                raw_acc_ii,
                                                                                crop_acc_ii,
                                                                                drop_acc_ii) + '\n'
                str_print += '--' * 18 + '\n'
                str_print += '{0:<13} {1:<7.3f} {2:<7.3f} {3:<7.3f}\n'.format('Loss', epoch_loss[0] / batches,
                                                                              epoch_loss[1] / batches,
                                                                              epoch_loss[2] / batches)
                log_string('Training Performance:\n')
                log_string(str_print)
                log_string('--' * 25)

                val_loss, best_val_loss, best_val_auc, eval_metrics = validate(data_loader=validate_loader,
                                                                               best_val_loss=best_val_loss,
                                                                               best_val_auc=best_val_auc,
                                                                               net=net,
                                                                               loss=loss,
                                                                               verbose=options.verbose)

                # write to tensorboard
                info = {'raw_loss': val_loss[0], 'crop_loss': val_loss[1], 'combined_loss': val_loss[2]}
                for tag, value in info.items():
                    val_logger.scalar_summary(tag, value, global_step)

                types_ = ['raw', 'crop', 'combined']
                for ii in range(len(types_)):
                    metrics_ = eval_metrics[ii]
                    for k, v in metrics_['aucs'].items():
                        val_logger.scalar_summary('{}_{}_auc'.format(conditions[k], types_[ii]), v, global_step)
                    for k, v in enumerate(metrics_['acc']):
                        val_logger.scalar_summary('{}_{}_accuracy'.format(conditions[k], types_[ii]), v, global_step)

                # save checkpoint model
                state_dict = net.module.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()

                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'val_loss': val_loss,
                    'eval_metrics': eval_metrics,
                    'save_dir': model_dir,
                    'state_dict': state_dict},
                    os.path.join(model_dir, '{}.ckpt'.format(global_step)))

            net.train()

        # end of this epoch
        scheduler.step()


@torch.no_grad()
def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    best_val_loss = kwargs['best_val_loss']
    best_val_auc = kwargs['best_val_auc']

    log_string('Validation Performance:\n')

    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    raw_loss, crop_loss, combined_loss = 0, 0, 0
    targets, raw_outputs, crop_outputs, combined_outputs = [], [], [], []

    # begin validation
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
            raw_outputs += [y_pred_raw]

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
            crop_outputs += [y_pred_crop]

            # final prediction
            y_pred = (y_pred_raw + y_pred_crop) / 2
            combined_outputs += [y_pred]
            targets += [y]

            # loss
            raw_loss += loss(y_pred_raw, y.float()).item()
            crop_loss += loss(y_pred_crop, y.float()).item()
            combined_loss += loss(y_pred, y.float()).item()

            # end of this batch
            batches += 1

    # metrics
    val_loss = np.array([raw_loss / batches, crop_loss / batches, combined_loss / batches])
    raw_eval_metrics = compute_metrics(torch.cat(raw_outputs).cpu(), torch.cat(targets).cpu())
    crop_eval_metrics = compute_metrics(torch.cat(crop_outputs).cpu(), torch.cat(targets).cpu())
    comb_eval_metrics = compute_metrics(torch.cat(combined_outputs).cpu(), torch.cat(targets).cpu())

    if val_loss[2] <= best_val_loss:
        best_val_loss = val_loss[2]
        loss_im_string = '(validation loss improved)'
    else:
        loss_im_string = ''

    # show information for this epoch
    str_print = ''
    str_print += '{0:<13}  {1:<6} {2:<7}    {3:<6} {4:<7}    {5:<6}   {6:<7}\n'.format('class_name',
                                                                                       'acc', 'AUC',
                                                                                       'acc', 'AUC',
                                                                                       'acc', 'AUC')
    str_print += '--' * 33 + '\n'
    for ii in range(len(raw_eval_metrics['acc'])):
        raw_acc_ii = round(raw_eval_metrics['acc'][ii], 2)
        raw_auc_ii = round(raw_eval_metrics['aucs'][ii], 2)
        crop_acc_ii = round(crop_eval_metrics['acc'][ii], 2)
        crop_auc_ii = round(crop_eval_metrics['aucs'][ii], 2)
        comb_acc_ii = round(comb_eval_metrics['acc'][ii], 2)
        comb_auc_ii = round(comb_eval_metrics['aucs'][ii], 2)
        str_print += '{0:<13} {1:<6}, {2:<7}   {3:<6}, {4:<7}   {5:<6}   {6:<7}'.format(conditions[ii],
                                                                                        str(raw_acc_ii),
                                                                                        str(raw_auc_ii)
                                                                                        , str(crop_acc_ii),
                                                                                        str(crop_auc_ii)
                                                                                        , str(comb_acc_ii),
                                                                                        str(comb_auc_ii)) + '\n'
    str_print += '--' * 33 + '\n'
    avg_acc = [raw_eval_metrics['acc'].mean(), crop_eval_metrics['acc'].mean(), comb_eval_metrics['acc'].mean()]
    str_print += '{0:<13}   {1:<6.3f}  \t\t\t{2:<7.3f} \t\t\t{3:<6.3f}\n'.format('Avg. Acc.', avg_acc[0],
                                                                                 avg_acc[1], avg_acc[2])

    str_print += '--' * 33 + '\n'
    avg_auc = [np.nanmean(np.array(list(raw_eval_metrics['aucs'].values()))),
               np.nanmean(np.array(list(crop_eval_metrics['aucs'].values()))),
               np.nanmean(np.array(list(comb_eval_metrics['aucs'].values())))]
    if avg_auc[2] >= best_val_auc:
        best_val_auc = avg_auc[2]
        auc_im_string = '(validation auc improved)'
    else:
        auc_im_string = ''

    str_print += '{0:<13}   {1:<6.3f}  \t\t\t{2:<7.3f} \t\t\t{3:<6.3f}   {4}\n'.format('Avg. AUC', avg_auc[0],
                                                                                       avg_auc[1],
                                                                                       avg_auc[2], auc_im_string)
    str_print += '--' * 33 + '\n'
    str_print += '{0:<13}   {1:<6.3f}  \t\t\t{2:<7.3f} \t\t\t{3:<6.3f}   {4}'.format('Loss', val_loss[0], val_loss[1],
                                                                                     val_loss[2], loss_im_string)
    log_string('\t\t\t\t  Raw \t\t\t  Crop \t\t\t Combined')
    log_string('--' * 33)
    log_string(str_print)
    log_string('--' * 33)
    log_string('')
    return val_loss, best_val_loss, best_val_auc, (raw_eval_metrics, crop_eval_metrics, comb_eval_metrics)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data_dir = options.data_dir

    ##################################
    # Initialize model
    ##################################
    image_size = (options.input_size, options.input_size)
    num_classes = len(target_conditions)
    num_attentions = options.num_attentions
    start_epoch = 0
    global_step = 0

    if options.model == 'densenet121':
        feature_net = densenet121(pretrained=True)
    elif options.model == 'resnet50':
        feature_net = resnet50(pretrained=True)
    elif options.model == 'inception':
        feature_net = inception_v3(pretrained=True)
    else:
        feature_net = inception_v3(pretrained=True)

    net = WSDAN_v2(num_classes=num_classes, M=num_attentions, K=options.K, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda"))

    if options.load_model:
        ckpt = options.load_model_path

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

    if options.weighted_loss:
        pos_weight = torch.from_numpy(compute_class_weights(train_dataset.labels, wt_type='balanced'))
    else:
        pos_weight = torch.ones([num_classes])
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

    train(global_step=global_step,
          data_loader=train_loader,
          net=net,
          feature_center=feature_center,
          loss=loss,
          optimizer=optimizer,
          model_dir=model_dir,
          verbose=options.verbose,
          val_freq=options.val_freq)
