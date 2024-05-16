import ml_collections


def get_general_configs():
    config = ml_collections.ConfigDict()
    # original
    config.devices = [4, 5, 6, 7]

    # data
    config.data = ml_collections.ConfigDict()

    # training
    config.training = ml_collections.ConfigDict()

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 50

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.weight_decay = 0.0
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42

    return config