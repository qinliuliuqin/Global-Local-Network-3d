from easydict import EasyDict as edict
from segmentation3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer


__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-segmentation pair list
__C.general.train_im_list = '/mnt/projects/CT_Dental/debug/crop/cropped_image.nii.gz/train.csv'

__C.general.val_im_list = '/mnt/projects/CT_Dental/debug/crop/cropped_image.nii.gz/test.csv'

# the output of training models and logs
__C.general.save_dir = '/mnt/projects/CT_Dental/debug/model_0805_2020'

# the model scale
__C.general.model_scale = 'fine'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training. Set to 0 if using cpu only.
__C.general.num_gpus = 0

# random seed used in training (debugging purpose)
__C.general.seed = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 4

# the resolution on which segmentation is performed
__C.dataset.spacing = [2.0, 2.0, 2.0]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [64, 64, 64]

__C.dataset.down_sample_ratio = 4

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
# 2) MASK: sampling crops randomly within segmentation mask
# 3) HYBRID: Sampling crops randomly with both GLOBAL and MASK methods
# 4) CENTER: sampling crops in the image center
__C.dataset.sampling_method = 'HYBRID'

# translation augmentation (unit: mm)
__C.dataset.random_translation = [0, 0, 0]

# spacing scale augmentation, spacing scale will be randomly selected from [min, max]
# during training, the image spacing will be spacing * scale
__C.dataset.random_scale = [1.0, 1.0]

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [AdaptiveNormalizer()]

##################################
# training loss
##################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity Loss which supports binary and multi-class segmentation
__C.loss.name = 'Focal'

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1] * __C.dataset.num_classes

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2

# the weights for global branch, local branch, and global-to-local branch
__C.loss.branch_weight = [0, 0, 1]

##################################
# net
##################################

__C.net = {}

# the network name
__C.net.name = 'global_local_net'

##################################
# training parameters
##################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 2001

# the number of samples in a batch
__C.train.batchsize = 1

# the number of threads for IO
__C.train.num_threads = 1

# the learning rate
__C.train.lr = 1e-4

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

# the number of batches to save model
__C.train.save_epochs = 2

__C.train.print_freq = 1

###################################
# debug parameters
###################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = False
