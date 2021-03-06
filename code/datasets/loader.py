import os
import PIL
import numpy as np
import torchvision.transforms as transforms
from copy import deepcopy

from datasets.class_sampler import ClassSampler
from datasets.cifar import CIFAR10Loader
from datasets.fashion_mnist import FashionMNISTLoader
from datasets.mnist_cluttered import ClutteredMNISTLoader
from datasets.mnist import MNISTLoader
from datasets.omniglot import OmniglotLoader
from datasets.permuted_mnist import PermutedMNISTLoader
from datasets.svhn import SVHNCenteredLoader, SVHNFullLoader
from datasets.utils import bw_2_rgb_lambda, resize_lambda, \
    simple_merger, sequential_test_set_merger

# ensures we get the same permutation
PERMUTE_SEED = 1

loader_map = {
    'mnist'        : MNISTLoader,
    'omniglot'     : OmniglotLoader,
    'permuted'     : PermutedMNISTLoader,
    'fashion'      : FashionMNISTLoader,
    'clutter'      : ClutteredMNISTLoader,
    'cifar10'      : CIFAR10Loader,
    'svhn'         : SVHNCenteredLoader,
    'svhn_centered': SVHNCenteredLoader,
    'svhn_full'    : SVHNFullLoader
}

def get_samplers(num_classes):
    ''' builds samplers taking into account previous classes'''
    test_samplers  = [ClassSampler(class_number=j, shuffle=False) for j in range(num_classes)]
    train_samplers = [ClassSampler(class_number=j, shuffle=True ) for j in range(num_classes)]
    return train_samplers, test_samplers


def get_rotated_loader(task, args, angles=[30, 70, 270], num_classes=10):
    ''' returns a list of loaders that are rotated by angles '''
    assert task in loader_map

    train_samplers, test_samplers = get_samplers(num_classes)
    rotation_transforms = [
        transforms.RandomRotation((a, a), resample=PIL.Image.BILINEAR) for a in angles
    ]

    loaders = [
        loader_map[task](path=args.data_dir, batch_size=args.batch_size,
                         train_sampler=tr, test_sampler=te,
                         transform=[rt], use_cuda=args.cuda)
        for rt in rotation_transforms
        for tr, te in zip(train_samplers, test_samplers)
    ]

    # append the normal loaders too
    args_clone      = deepcopy(args)
    args_clone.task = [task]
    loaders.extend(get_split_data_loaders(args_clone, num_classes))

    print("total rotated loaders: {}".format(loaders))
    return loaders

def get_loader(args, transform=None, target_transform=None,
               train_sampler=None, test_sampler=None,
               increment_seed=True, sequentially_merge_test=True):
    ''' increment_seed: increases permutation rng seed,
        sequentially_merge_test: merge all the test sets sequentially '''
    task = args.task
    global PERMUTE_SEED
    print("few_sort learning dataset is loading")
    
    # overwrite data dir for fashion MNIST because it has issues being
    # in the same directory as regular MNIST
    if task == 'fashion':
        data_dir = os.path.join(args.data_dir, "fashion")
    else:
        data_dir = args.data_dir
    if args.few_sort:
        if task == 'mnist': # same few sort can be implemented in other dataset also
            kwargs = {}
            return loader_map[task](path=args.data_dir, batch_size=args.batch_size,few_sort=args.few_sort,num_of_sample=args.num_sample,
                                transform=transform, target_transform=target_transform,
                                train_sampler=train_sampler, test_sampler=test_sampler,
                                use_cuda=args.cuda, **kwargs)
    
    if '+' in task:  # the merge operand
        loader = []
        for split in task.split('+'):
            args_clone      = deepcopy(args)
            args_clone.task = split

            # XXX: create transforms to resize and convert to rgb
            # this should be parameterized in the future
            transform_list = [
                transforms.Resize((32, 32)), # XXX: parameterize
                transforms.Lambda(lambda img: bw_2_rgb_lambda(img))
            ]
            if transform is not None and isinstance(transform, list):
                transform_list = transform + transform_list

            loader.append(get_loader(args_clone, transform=transform))
            if increment_seed:
                PERMUTE_SEED += 1

        PERMUTE_SEED = 1
        if sequentially_merge_test:
            loader = sequential_test_set_merger(loader)
    elif 'rotated' in task:
        task = task.split('_')[1]
        loader = get_rotated_loader(task, args)
        if sequentially_merge_test:
            loader = sequential_test_set_merger(loader)
    else:
        assert task in loader_map, "unknown task requested"
        kwargs = {}
        if task == 'permuted':
            kwargs['seed'] = PERMUTE_SEED

        return loader_map[task](path=data_dir, batch_size=args.batch_size,
                                transform=transform, target_transform=target_transform,
                                train_sampler=train_sampler, test_sampler=test_sampler,
                                use_cuda=args.cuda, **kwargs)

    return loader


def _check_for_sublist(loaders):
    is_sublist = False
    for l in loaders:
        if isinstance(l, list):
            is_sublist = True
            break

    return is_sublist


def get_split_data_loaders(args, num_classes, transform=None, target_transform=None):
    ''' helper to return the model and the loader '''
    # we build 10 samplers as all of the below have 10 classes
    train_samplers, test_samplers = get_samplers(num_classes=num_classes)
    global PERMUTE_SEED

    loaders = []
    if '+' in args.task:  # the merge operand
        for split in args.task.split('+'):
            args_clone      = deepcopy(args)
            args_clone.task = split
            loaders.extend([
                get_loader(args_clone, transform=transform,
                           target_transform=target_transform,
                           train_sampler=tr, test_sampler=te,
                           increment_seed=False, sequentially_merge_test=False)
                for tr, te in zip(train_samplers, test_samplers)
            ])
            PERMUTE_SEED += 1
    else:
        # iterate over samplers and generate
        loaders = [
            get_loader(args, transform=transform,
                       target_transform=target_transform,
                       train_sampler=tr, test_sampler=te,
                       increment_seed=False)
            for tr, te in zip(train_samplers, test_samplers)
        ]

    if _check_for_sublist(loaders):
        loaders = [item for sublist in loaders for item in sublist]

    PERMUTE_SEED = 1
    return sequential_test_set_merger(loaders)
