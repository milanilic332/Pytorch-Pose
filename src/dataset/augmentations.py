import imgaug as ia
from imgaug import augmenters as iaa


def get_augmentations():
    seq = iaa.Sequential([
                          iaa.Sometimes(0.25, iaa.Add((-25, 25))),
                          iaa.Sometimes(0.25, iaa.Multiply((0.8, 1.2))),
                          iaa.Fliplr(0.5),
                         ], random_order=True)

    def activator(images, augmenter, parents, default):
        if augmenter.name in ['UnnamedMultiply', 'UnnamedAdd']:
            return False
        else:
            return default

    hooks = ia.HooksImages(activator=activator)

    return seq, hooks
