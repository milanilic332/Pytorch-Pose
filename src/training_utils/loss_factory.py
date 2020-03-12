import torch.nn as nn


def build_losses(config):
    loss_cfg = config['loss']

    paf_loss_name = loss_cfg['paf_loss']
    class_loss_name = loss_cfg['class_loss']

    paf_loss = eval('nn.{}()'.format(paf_loss_name))
    class_loss = eval('nn.{}()'.format(class_loss_name))

    return paf_loss, class_loss