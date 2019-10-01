"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os

from torchvision.models import densenet121

from utils.other_utils import compute_class_weights, compute_metrics, visualize_attention, generate_title

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models import *
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


@torch.no_grad()
def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    viz_dir = kwargs['viz_dir']
    loss = kwargs['loss']

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
        for i, (X, y, img_names) in enumerate(data_loader):

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
            upsampled_attention_map = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3)))
            crop_mask = upsampled_attention_map > theta_c
            crop_images = []
            box_coords = torch.zeros(options.batch_size, 4)
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()

                # visualize the attention map
                box_coords[batch_index] = torch.FloatTensor([height_min, height_max, width_min, width_max])

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

            # Visualize attention and crops
            for batch_index in range(crop_mask.size(0)):
                visualize_attention(X[batch_index].clone().cpu().numpy(),
                                    box_coords[batch_index], upsampled_attention_map[batch_index].clone().cpu().numpy(),
                                    title=generate_title(y[batch_index], y_pred[batch_index],
                                                         np.array(conditions)[target_conditions]),
                                    img_save_path=os.path.join(viz_dir, img_names[batch_index] + '.png'))

    # metrics
    val_loss = np.array([raw_loss / batches, crop_loss / batches, combined_loss / batches])
    raw_eval_metrics = compute_metrics(torch.cat(raw_outputs).cpu(), torch.cat(targets).cpu())
    crop_eval_metrics = compute_metrics(torch.cat(crop_outputs).cpu(), torch.cat(targets).cpu())
    comb_eval_metrics = compute_metrics(torch.cat(combined_outputs).cpu(), torch.cat(targets).cpu())

    # show information for this epoch
    str_print = ''
    str_print += '{0:<13}  {1:<6} {2:<7}    {3:<6} {4:<7}    {5:<6}   {6:<7}\n'.format('class_name',
                                                                                       'acc', 'AUC',
                                                                                       'acc', 'AUC',
                                                                                       'acc', 'AUC')
    str_print += '--' * 33 + '\n'
    for ii in range(len(raw_eval_metrics['acc'])):
        raw_acc_ii = round(raw_eval_metrics['acc'][ii], 2)
        raw_auc_ii = round(raw_eval_metrics['aucs'][ii], 3)
        crop_acc_ii = round(crop_eval_metrics['acc'][ii], 2)
        crop_auc_ii = round(crop_eval_metrics['aucs'][ii], 3)
        comb_acc_ii = round(comb_eval_metrics['acc'][ii], 2)
        comb_auc_ii = round(comb_eval_metrics['aucs'][ii], 3)
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

    str_print += '{0:<13}   {1:<6.3f}  \t\t\t{2:<7.3f} \t\t\t{3:<6.3f}\n'.format('Avg. AUC', avg_auc[0],
                                                                                 avg_auc[1],
                                                                                 avg_auc[2])
    str_print += '--' * 33 + '\n'
    str_print += '{0:<13}   {1:<6.3f}  \t\t\t{2:<7.3f} \t\t\t{3:<6.3f} '.format('Loss', val_loss[0], val_loss[1],
                                                                                val_loss[2])
    log_string('\t\t\t\t  Raw \t\t\t  Crop \t\t\t Combined')
    log_string('--' * 33)
    log_string(str_print)
    log_string('--' * 33)
    log_string('')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data_dir = options.data_dir

    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    iter_num = options.load_model_path.split('/')[-1].split('.')[0]

    img_dir = os.path.join(save_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    viz_dir = os.path.join(img_dir, iter_num)
    viz_dir_train = os.path.join(viz_dir, 'train')
    viz_dir_test = os.path.join(viz_dir, 'test')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        os.makedirs(viz_dir_train)
        os.makedirs(viz_dir_test)

    # bkp of inference procedure
    os.system('cp {}/inference_new.py {}'.format(BASE_DIR, save_dir))

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    ##################################
    # Initialize model
    ##################################
    image_size = (options.input_size, options.input_size)
    num_classes = len(target_conditions)
    num_attentions = options.num_attentions

    if options.model == 'densenet121':
        feature_net = densenet121(pretrained=True)
    elif options.model == 'resnet152':
        feature_net = resnet152(pretrained=True)
    elif options.model == 'inception':
        feature_net = inception_v3(pretrained=True)
    else:
        feature_net = inception_v3(pretrained=True)

    net = WSDAN_v2(num_classes=num_classes, M=num_attentions, K=options.K, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    # feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda"))
    feature_center = torch.zeros(2, num_classes, num_attentions, net.num_features * net.expansion).to(
        torch.device("cuda"))

    # Load ckpt and get state_dict
    ckpt = options.load_model_path
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
    # TESTING
    ##################################

    log_string('')
    log_string('Start Testing: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(validate_dataset)))

    validate(data_loader=validate_loader,
             viz_dir=viz_dir_test,
             net=net,
             loss=loss,
             verbose=options.verbose)
