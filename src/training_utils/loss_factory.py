import torch.nn as nn


def build_losses(config):
    """Creates losses for paf and keypoint outputs

    :param config:          training config object
    :return:                paf loss, class loss
    """
    loss_cfg = config['loss']

    paf_loss_name = loss_cfg['paf_loss']
    class_loss_name = loss_cfg['class_loss']

    try:
        if hasattr(nn, paf_loss_name):
            paf_loss = eval('nn.{}()'.format(paf_loss_name))
        else:
            paf_loss = eval('{}()'.format(paf_loss_name))

        if hasattr(nn, class_loss_name):
            class_loss = eval('nn.{}()'.format(class_loss_name))
        else:
            class_loss = eval('{}()'.format(class_loss_name))
    except:
        raise ValueError('Can\'t load loss function.')

    return paf_loss, class_loss
