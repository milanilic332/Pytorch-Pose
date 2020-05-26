import imgaug as ia
from imgaug import augmenters as iaa


def get_augmentations():
    seq = iaa.Sequential([
                          iaa.Sometimes(0.5, iaa.Add((-10, 10))),
                          iaa.Sometimes(0.5, iaa.Multiply((0.9, 1.1))),
                          iaa.Fliplr(0.5),
                          iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                                        order=[0, 1])),
                         ], random_order=True)

    return seq
