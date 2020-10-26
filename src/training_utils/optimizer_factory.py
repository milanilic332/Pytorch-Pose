import torch.optim as optim
import inspect


def build_optimizer(model_params, config):
    """

    :param model_params:        model parameters (first optimizer arg)
    :param config:              training config object
    :return:                    optimizer object
    """
    optim_cfg = config['optimizer']

    optim_name = optim_cfg['name']

    optim_params = list(inspect.signature(eval('optim.{}'.format(optim_name)).__init__).parameters)[1:]

    args = [f'{param}={optim_cfg[param]}' for param in optim_params if optim_cfg.get(param)]

    try:
        if hasattr(optim, optim_name):
            optimizer = eval('optim.{}(params=model_params, {})'.format(optim_name, *args))
        else:
            optimizer = eval('{}(params=model_params, {})'.format(optim_name, *args))
    except:
        raise ValueError('Can\'t load optimizer.')

    return optimizer
