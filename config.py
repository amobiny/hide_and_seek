from optparse import OptionParser


parser = OptionParser()

parser.add_option('-v', '--verbose', dest='verbose', default=100, type='int',
                  help='show information for each <verbose> iterations (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=2000, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')

parser.add_option('-j', '--workers', dest='workers', default=1, type='int',
                  help='number of data loading workers (default: 16)')
parser.add_option('-e', '--epochs', dest='epochs', default=300, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=8, type='int',
                  help='batch size (default: 16)')

# data and model
parser.add_option('--dd', '--data_dir', dest='data_dir',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/CheXpert-v1.0-small',
                  help='path to the data directory')
parser.add_option('--sz', '--input_size', dest='input_size', default=448, type='int',
                  help='desired input size (default: 448)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=14, type='int',
                  help='number of classes (default: 200)')
parser.add_option('--natt', '--num_attentions', dest='num_attentions', default=10, type='int',
                  help='desired number of attention maps (default: 32)')
parser.add_option('--katt', '--K', dest='K', default=4, type='int',
                  help='The number of attention maps selected randomly in the training phase (default: 4)')
parser.add_option('--dl', '--data_len', dest='data_len', default=None, type='int',
                  help='Length of the training and vaidation data (default: None which takes all)')
parser.add_option('--mdl', '--model', dest='model', default='inception',
                  help='name of the model; densenet121 or resnet50 or inception (default: densenet121)')

# loss
parser.add_option('--lr', '--learning-rate', dest='lr', default=1e-3, type='float',
                  help='learning rate (default: 1e-3)')
parser.add_option('--lm', '--weighted_loss', dest='weighted_loss', default=True,
                  help='whether to use weighted loss or not')

parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')


# loading a trained model
parser.add_option('--wl', '--load_model', dest='load_model', default=False,
                  help='load checkpoint model (default: False)')
parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/hide_and_seek/'
                          'save/20190916_142925/models/014.ckpt',
                  help='path to load a .ckpt model')

(options, args) = parser.parse_args()
