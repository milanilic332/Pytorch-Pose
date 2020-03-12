import torch.optim as optim
import inspect


def build_optimizer(model_params, config):
    optim_cfg = config['optimizer']

    optim_name = optim_cfg['name']

    optim_params = list(inspect.signature(eval('optim.{}'.format(optim_name)).__init__).parameters)[1:]

    args = [f'{param}={optim_cfg[param]}' for param in optim_params if optim_cfg.get(param)]

    try:
        optimizer = eval('optim.{}(params=model_params, {})'.format(optim_name, *args))
    except:
        raise ValueError('Can\'t load optimizer.')

    return optimizer