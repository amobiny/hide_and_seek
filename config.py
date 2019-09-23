from optparse import OptionParser


parser = OptionParser()
parser.add_option('-j', '--workers', dest='workers', default=1, type='int',
                  help='number of data loading workers (default: 16)')
parser.add_option('-e', '--epochs', dest='epochs', default=80, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                  help='batch size (default: 16)')
parser.add_option('-v', '--verbose', dest='verbose', default=100, type='int',
                  help='show information for each <verbose> iterations (default: 100)')
parser.add_option('--sz', '--input_size', dest='input_size', default=448, type='int',
                  help='desired input size (default: 448)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=14, type='int',
                  help='number of classes (default: 200)')
parser.add_option('--natt', '--num_attentions', dest='num_attentions', default=32, type='int',
                  help='desired number of attention maps (default: 32)')
parser.add_option('--katt', '--K', dest='K', default=4, type='int',
                  help='The number of attention maps selected randomly in the training phase (default: 4)')
parser.add_option('--lr', '--learning-rate', dest='lr', default=1e-3, type='float',
                  help='learning rate (default: 1e-3)')
parser.add_option('--sf', '--save-freq', dest='save_freq', default=1, type='int',
                  help='saving frequency of .ckpt models (default: 1)')
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')
parser.add_option('--init', '--initial-training', dest='initial_training', default=1, type='int',
                  help='train from 1-beginning or 0-resume training (default: 1)')
parser.add_option('--dn', '--data_name', dest='data_name', default='cheXpert',
                  help='name of the data; CUB or cheXpert (default: CUB)')
parser.add_option('--dl', '--data_len', dest='data_len', default=None, type='int',
                  help='Length of the training and vaidation data (default: None which takes all)')

# loading a trained model
parser.add_option('--lm', '--load_model', dest='load_model', default=False,
                  help='load checkpoint model (default: False)')
parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/hide_and_seek/'
                          'save/20190916_142925/models/014.ckpt',
                  help='path to load a .ckpt model')

(options, args) = parser.parse_args()
