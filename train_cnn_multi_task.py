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
from models import *
from datetime import datetime
from utils.other_utils import compute_class_weights
from utils.logger_utils import Logger
from config import options

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
    correct_pred = target.eq(output.round().long())
    accuracy = torch.sum(correct_pred, dim=0)
    return accuracy.cpu().numpy() * (100. / batch_size)


def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    global_step = kwargs['global_step']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    epoch = kwargs['epoch']
    save_freq = kwargs['save_freq']
    model_dir = kwargs['model_dir']
    verbose = kwargs['verbose']

    # metrics initialization
    batches = 0
    epoch_loss = np.array([0], dtype='float')
    epoch_acc = np.zeros((1, num_classes), dtype='float')

    # begin training
    start_time = time.time()
    log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()
    for i, (X, y) in enumerate(data_loader):
        batch_start = time.time()
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

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc += accuracy_generator(y_pred, y)

        # end of this batch
        batches += 1
        batch_end = time.time()
        if (i + 1) % verbose == 0:
            acc_str = ''
            for ii in range(epoch_acc.shape[1]):
                acc_ii = round(epoch_acc[0, ii] / batches, 2)
                acc_str += conditions[ii] + ': ' + str(acc_ii) + ',\t'
            log_string('\tBatch {0}, Time {1:3.2f}, Loss {2:.4f}, meanAcc {3:2.2f}'.format(i + 1,
                                                                                           batch_end - batch_start,
                                                                                           epoch_loss[0] / batches,
                                                                                           np.mean(
                                                                                               epoch_acc) / batches))
            # log_string('\tAccuracy: ' + acc_str)
            info = {'loss': epoch_loss[0] / batches}
            for tag, value in info.items():
                train_logger.scalar_summary(tag, value, global_step)

    # save checkpoint model
    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': model_dir,
            'state_dict': state_dict},
            os.path.join(model_dir, '%03d.ckpt' % (epoch + 1)))

    # end of this epoch
    end_time = time.time()

    # show information for this epoch
    log_string('--' * 25)
    acc_str = ''
    for ii in range(epoch_acc.shape[1]):
        acc_ii = round(epoch_acc[0, ii] / batches, 2)
        acc_str += conditions[ii] + ': ' + str(acc_ii) + ',\t'
    log_string('\tEpoch {0}, Time {1:3.2f}, Loss {2:.4f}, meanAcc {3:2.2f}'.format(epoch,
                                                                                   end_time - start_time,
                                                                                   epoch_loss[0] / batches,
                                                                                   np.mean(epoch_acc) / batches))
    log_string('\tAccuracy: ' + acc_str)

    return global_step


def validate(**kwargs):
    # Retrieve training configuration
    global_step = kwargs['global_step']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']
    log_string('--' * 25)
    log_string('Running Validation ... ')

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0] * options.num_classes, dtype='float')

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
            y_pred = net(X)

            epoch_loss += loss(y_pred, y.float()).item()

            # metrics: top-1, top-3, top-5 error
            epoch_acc += accuracy_generator(y_pred, y)

            # end of this batch
            batches += 1

    # end of validation
    end_time = time.time()

    # show information for this epoch
    acc_str = ''
    for ii in range(len(epoch_acc)):
        acc_ii = round(epoch_acc[ii] / batches, 2)
        acc_str += conditions[ii] + ': ' + str(acc_ii) + ',\t'
    log_string('\tTime {0:3.2f}, Loss {1:.4f}, meanAcc {2:2.2f}'.format(end_time - start_time,
                                                                        epoch_loss / batches,
                                                                        np.mean(epoch_acc) / batches))
    log_string('\tAccuracy: ' + acc_str)
    log_string('--' * 25)
    log_string('')

    info = {'loss': epoch_loss / batches}
    for tag, value in info.items():
        val_logger.scalar_summary(tag, value, global_step)

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

    net = resnet50(pretrained=True)
    net.aux_logits = False
    # Replace the top layer for finetuning.
    net.fc = nn.Linear(net.fc.in_features, num_classes)

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
    optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00001)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.float().to(torch.device("cuda")))

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
    num_train_batches = len(train_loader) / options.batch_size

    global_step = 0

    for epoch in range(start_epoch, options.epochs):
        global_step = train(epoch=epoch,
                            global_step=global_step,
                            data_loader=train_loader,
                            net=net,
                            loss=loss,
                            optimizer=optimizer,
                            save_freq=options.save_freq,
                            model_dir=model_dir,
                            verbose=options.verbose)
        val_loss = validate(global_step=global_step,
                            data_loader=validate_loader,
                            net=net,
                            loss=loss,
                            verbose=options.verbose)
        scheduler.step()
