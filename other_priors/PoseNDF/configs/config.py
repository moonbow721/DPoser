import yaml
def load_config(path):
    """ load config file"""
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def edit_config():
    """update configs"""
    pass