"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.models import densenet121

from models import *
from datetime import datetime
from utils.other_utils import compute_class_weights, compute_metrics
from utils.logger_utils import Logger
from config import options
from dataset.chexpert_dataset import CheXpertDataSet as data

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

conditions = ['No Find', 'Enlgd Card.', 'Crdmgly', 'Opcty', 'Lsn', 'Edma', 'Cnsldton',
              'Pnumn', 'Atlctss', 'Pnmthrx', 'Plu. Eff.', 'Plu. Othr', 'Frctr', 'S. Dev.']
target_conditions = None


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
    sigmoid_output = torch.sigmoid(output)
    correct_pred = target.eq(sigmoid_output.round().long())
    accuracy = torch.sum(correct_pred, dim=0)
    return accuracy.cpu().numpy() * (100. / batch_size)


def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    global_step = kwargs['global_step']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    model_dir = kwargs['model_dir']
    verbose = kwargs['verbose']
    val_freq = kwargs['val_freq']

    best_val_loss = 100
    for epoch in range(start_epoch, options.epochs):

        # metrics initialization
        batches = 0
        epoch_loss = np.array([0], dtype='float')
        epoch_acc = np.zeros((1, num_classes), dtype='float')

        # begin training
        start_time = time.time()
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()
        for i, (X, y) in enumerate(data_loader):
            global_step += 1
            # obtain data for training
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

            y_pred = net(X)

            # loss
            batch_loss = loss(y_pred, y.float())
            epoch_loss += batch_loss.item()

            # backward
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_acc += accuracy_generator(y_pred, y)

            # end of this batch
            batches += 1

            if (i + 1) % verbose == 0:

                log_string('Batch {0}, Loss {1:.4f}, meanAcc {2:2.2f}'.format(i + 1,
                                                                              epoch_loss[0] / batches,
                                                                              np.mean(epoch_acc) / batches))
                info = {'loss': epoch_loss[0] / batches}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)

            if (i + 1) % val_freq == 0:
                # results on training data
                acc_str = ''
                for ii in range(epoch_acc.shape[1]):
                    acc_ii = round(epoch_acc[0, ii] / batches, 2)
                    acc_str += '{}: '.format(ii) + str(acc_ii) + ',\t'
                log_string('Training Accuracy:')
                log_string(acc_str)

                val_loss, best_val_loss, eval_metrics = validate(data_loader=validate_loader,
                                                                 best_val_loss=best_val_loss,
                                                                 net=net,
                                                                 loss=loss,
                                                                 verbose=options.verbose)

                # write to tensorboard
                info = {'loss': epoch_loss, 'lr': optimizer.param_groups[0]['lr']}
                for tag, value in info.items():
                    val_logger.scalar_summary(tag, value, global_step)
                for k, v in eval_metrics['aucs'].items():
                    val_logger.scalar_summary('{}_auc'.format(conditions[k]), v, global_step)
                for k, v in enumerate(eval_metrics['acc']):
                    val_logger.scalar_summary('{}_accuracy'.format(conditions[k]), v, global_step)

                # save checkpoint model
                state_dict = net.module.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()

                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'val_loss': val_loss,
                    'avg_acc': eval_metrics['acc'].mean(),
                    'avg_aucs': np.nanmean(np.array(list(eval_metrics['aucs'].values()))),
                    'save_dir': model_dir,
                    'state_dict': state_dict},
                    os.path.join(model_dir, '{}.ckpt'.format(global_step)))

            net.train()

        # end of this epoch
        end_time = time.time()
        scheduler.step()

        # show information for this epoch
        log_string('--' * 25)
        log_string('Epoch {0}, Time {1:3.2f}'.format(epoch, end_time - start_time))


@torch.no_grad()
def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    best_val_loss = kwargs['best_val_loss']
    log_string('--' * 25)
    log_string('Running Validation ... ')

    # metrics initialization
    batches = 0
    epoch_loss = 0
    targets, outputs = [], []

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))
            y_pred = net(X)
            epoch_loss += loss(y_pred, y.float()).item()
            outputs += [y_pred]
            targets += [y]
            batches += 1

    # end of validation
    end_time = time.time()
    epoch_loss /= batches
    eval_metrics = compute_metrics(torch.cat(outputs).cpu(), torch.cat(targets).cpu())

    if epoch_loss <= best_val_loss:
        best_val_loss = epoch_loss
        im_string = '(validation loss improved)'
    else:
        im_string = ''
    # show information for this epoch
    str_print = ''
    for ii in range(len(eval_metrics['acc'])):
        acc_ii = round(eval_metrics['acc'][ii], 2)
        auc_ii = round(eval_metrics['aucs'][ii], 2)
        str_print += '{0:<13}: {1:<7}, {2:<7}'.format(conditions[ii], str(acc_ii), str(auc_ii)) + '\n'
    log_string('Time {0:3.2f}, Loss {1:.4f}, meanAcc {2:2.2f} {3}'.format(end_time - start_time,
                                                                          epoch_loss,
                                                                          eval_metrics['acc'].mean(),
                                                                          im_string))
    log_string('{0:<13} {1:<9} {2:<10}'.format('Class Name', 'Accuracy', 'AUC'))
    log_string(str_print)
    log_string('--' * 25)
    log_string('')

    return epoch_loss, best_val_loss, eval_metrics


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/CheXpert-v1.0-small'

    ##################################
    # Initialize model
    ##################################
    image_size = (options.input_size, options.input_size)
    num_classes = options.num_classes
    num_attentions = options.num_attentions
    start_epoch = 0
    global_step = 0

    if options.model == 'densenet121':
        net = densenet121(pretrained=True)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs, num_classes)
    elif options.model == 'resnet50':
        net = resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif options.model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    # Replace the top layer for finetuning.

    if options.load_model:
        ckpt = options.ckpt

        # Load ckpt and get state_dict
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']
        global_step = checkpoint['global_step']

        # Load weights
        net.load_state_dict(state_dict)
        log_string('Network loaded from {}'.format(options.ckpt))

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

    # bkp of train procedure
    os.system('cp {}/train_cnn_multi_task.py {}'.format(BASE_DIR, save_dir))
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

    if options.weighted_loss:
        pos_weight = torch.from_numpy(compute_class_weights(train_dataset.labels, wt_type='balanced'))
    else:
        pos_weight = torch.ones([num_classes])
    # optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00001)
    optimizer = torch.optim.Adam(net.parameters(), lr=options.lr)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.float().to(torch.device("cuda")))

    ##################################
    # Learning rate scheduling
    ##################################
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

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
    num_train_batches = len(train_loader) / options.batch_size

    train(global_step=global_step,
          data_loader=train_loader,
          net=net,
          loss=loss,
          optimizer=optimizer,
          model_dir=model_dir,
          verbose=options.verbose,
          val_freq=options.val_freq)
