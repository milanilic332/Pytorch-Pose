from torch.utils.data import DataLoader

from src.dataset.datasets import PoseDataset


def build_dataloaders(config):
    train_dataset = PoseDataset(augment=True)

    train_dataloader = DataLoader(train_dataset,
                                  config['default']['batch_size'],
                                  shuffle=True,
                                  num_workers=8)

    if config['default'].get('validation'):
        val_dataset = PoseDataset(augment=False)

        val_dataloader = DataLoader(val_dataset,
                                    config['default']['batch_size'],
                                    shuffle=False,
                                    num_workers=8)

    return [train_dataloader, val_dataloader if config['default'].get('validation') else None]
