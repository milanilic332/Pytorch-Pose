import inspect

from src.networks.resnet_kp import ResNetKP
from src.networks.hourglass_v1 import HourglassV1


def build_network(config):
    network_cfg = config['network']

    network_name = network_cfg['name']

    network_params = list(inspect.signature(eval(network_name).__init__).parameters)[1:]

    args = [f'{param}={network_cfg[param]}' for param in network_params if network_cfg.get(param)]

    try:
        model = eval('{}({})'.format(network_name, ', '.join(args)))
    except:
        raise ValueError('Can\'t load network.')

    return model.cuda()

