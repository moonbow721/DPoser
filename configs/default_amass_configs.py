import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # original
    config.OUTPUT_DIR = 'output'
    config.DATASET = ml_collections.ConfigDict()
    config.DATASET.TRAIN_DATASET = 'amass'
    config.DATASET.TEST_DATASET = 'amass'
    config.DATASET.HYBRID_JOINTS_TYPE = ''

    # data
    config.data = data = ml_collections.ConfigDict()
    data.normalize = True
    data.rot_rep = 'axis'  # rot6d or axis
    data.min_max = False  # Z-score or min-max Normalize

    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 1280
    training.n_iters = 400001
    training.log_freq = 50
    training.eval_freq = 50000
    training.save_freq = 50000
    training.auxiliary_loss = False     # not recommended
    training.denoise_steps = 10  # for computing auxiliary loss
    training.render = False  # save obj files and render results while validating
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 50
    evaluate.num_samples = 500  # generation setting

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
