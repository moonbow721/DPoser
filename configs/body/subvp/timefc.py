from configs.body.default_amass_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'subvpsde'
    training.continuous = True

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # model
    model = config.model
    model.type = 'TimeFC'
    model.HIDDEN_DIM = 1024
    model.EMBED_DIM = 512
    model.N_BLOCKS = 2
    model.dropout = 0.1
    model.fourier_scale = 16
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nonlinearity = 'swish'
    model.embedding_type = 'positional'    # Or 'fourier'

    return config
