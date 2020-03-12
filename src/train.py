import configparser

from src.networks.network_factory import build_network
from src.dataset.dataloader_factory import build_dataloaders
from src.training_utils.optimizer_factory import build_optimizer
from src.training_utils.loss_factory import build_losses


def main(config):
    model = build_network(config)
    print(model)

    optimizer = build_optimizer(model.parameters(), config)
    print(optimizer)

    paf_loss, class_loss = build_losses(config)
    print(paf_loss, class_loss)

    train_dataloader, val_dataloader = build_dataloaders(config)
    print(train_dataloader, val_dataloader)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./train.cfg')

    main(config)
