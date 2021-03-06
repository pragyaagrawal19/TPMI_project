from __future__ import print_function
import torch
from torchvision import datasets, transforms

from datasets.svhn_full import SVHNFull
from datasets.utils import create_loader


class SVHNFullLoader(object):
    ''' This loads the original SVHN dataset (non-centered).

        The classes here are BCE classes where each bit
        signifies the presence of the digit in the img'''

    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # first grab the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform, target_transform)

        # build the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset, train_sampler, batch_size,
                                          shuffle=True if train_sampler is None else False, **kwargs)

        self.test_loader = create_loader(test_dataset, test_sampler, batch_size,
                                         shuffle=False, **kwargs)

        self.output_size = 10
        self.batch_size  = batch_size
        self.img_shp     = [3, 32, 32]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        transform_list = []
        if transform:
            assert isinstance(transform, list)
            transform_list = list(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = SVHNFull(path, split='train', download=True,
                                 transform=transforms.Compose(transform_list),
                                 target_transform=target_transform)
        test_dataset = SVHNFull(path, split='test', download=True,
                                transform=transforms.Compose(transform_list),
                                target_transform=target_transform)
        return train_dataset, test_dataset


class SVHNCenteredLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # first grab the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform, target_transform)

        # build the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset, train_sampler, batch_size,
                                          shuffle=True if train_sampler is None else False, **kwargs)

        self.test_loader = create_loader(test_dataset, test_sampler, batch_size,
                                         shuffle=False, **kwargs)

        self.output_size = 10
        self.batch_size  = batch_size
        self.img_shp     = [3, 32, 32]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        transform_list = []
        if transform:
            assert isinstance(transform, list)
            transform_list = list(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = datasets.SVHN(path, split='train', download=True,
                                      transform=transforms.Compose(transform_list),
                                      target_transform=target_transform)
        test_dataset = datasets.SVHN(path, split='test', download=True,
                                     transform=transforms.Compose(transform_list),
                                     target_transform=target_transform)
        return train_dataset, test_dataset
