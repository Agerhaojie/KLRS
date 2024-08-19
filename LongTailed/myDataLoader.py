import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from myDatasets import IMBALANCECIFAR10, IMBALANCECIFAR100, TINYIMAGENET, get_cls_num_list, LT, INAT
from torch.utils.data.sampler import Sampler

import os
import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from myDatasets import IMBALANCECIFAR10, IMBALANCECIFAR100, TINYIMAGENET, get_cls_num_list, LT, INAT

RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'ImageNet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}



def get_train_val_test_loader(args, train_sampler = None):
    print("==============================================>", train_sampler is None)
    if train_sampler is not None:

        sampler_dic = {
            'sampler': get_sampler(),
            'params': {'num_samples_cls': 4}
            }
    else:
        sampler_dic = None

    test_loader = None
    torch.manual_seed(args.seed)


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



    cifar_root = "./data/"

    if args.samplebylabel==False:
        if args.dataset == 'cifar10':
            train_dataset = IMBALANCECIFAR10(args= args, root = cifar_root, imb_type=args.imb_type,
                                            imb_factor=args.imb_factor,
                                            rand_number=args.seed, train=True, download=True,
                                            transform=transform_train, bylabel = False)

            val_dataset = datasets.CIFAR10(root = cifar_root, train=False, download=True, transform=transform_val)
        elif args.dataset == 'cifar100':
            train_dataset = IMBALANCECIFAR100(args=args,root = cifar_root, imb_type=args.imb_type,
                                            imb_factor=args.imb_factor,
                                            rand_number=args.seed, train=True, download=True,
                                            transform=transform_train, bylabel = False)
            val_dataset = datasets.CIFAR100(root = cifar_root, train=False, download=True, transform=transform_val)

    else:
        if args.dataset == 'cifar10':
            train_dataset = IMBALANCECIFAR10(args= args, root = cifar_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                            rand_number=args.seed, train=True, download=True,
                                            transform=transform_train, bylabel = True)

            val_dataset = datasets.CIFAR10(root = cifar_root, train=False, download=True, transform=transform_val)
        elif args.dataset == 'cifar100':
            train_dataset = IMBALANCECIFAR100(args=args,root = cifar_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                            rand_number=args.seed, train=True, download=True,
                                            transform=transform_train, bylabel=True)
            val_dataset = datasets.CIFAR100(root = cifar_root, train=False, download=True, transform=transform_val)


    if args.dataset=='cifar10':
        cls_num_list = train_dataset.get_cls_num_list(10)
    elif args.dataset=='cifar100':
        cls_num_list = train_dataset.get_cls_num_list(100)
    args.cls_num_list = cls_num_list

    train_sampler = None

    if args.samplebylabel==False:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.labelbatch_size*args.subbatch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=None if train_sampler is None else train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.labelbatch_size*args.subbatch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.labelbatch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=None if train_sampler is None else train_sampler, collate_fn = collate_func)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.labelbatch_size*args.subbatch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    if args.dataset =='cifar10':
        trainval_dataset = IMBALANCECIFAR10(args= args, root = cifar_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                rand_number=args.seed, train=True, download=True,
                                                transform=transform_train, bylabel = False)

        trainval_loader = torch.utils.data.DataLoader(
                trainval_dataset, batch_size=args.labelbatch_size*args.subbatch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=None if train_sampler is None else train_sampler)
    elif args.dataset=='cifar100':
        trainval_dataset = IMBALANCECIFAR100(args= args, root = cifar_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                                rand_number=args.seed, train=True, download=True,
                                                transform=transform_train, bylabel = False)

        trainval_loader = torch.utils.data.DataLoader(
                trainval_dataset, batch_size=args.labelbatch_size*args.subbatch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=None if train_sampler is None else train_sampler)
    #print('len data loader', len(train_loader), len(val_loader))

    num_training = len(train_dataset.targets)

    return train_loader, val_loader, test_loader, trainval_loader

def get_data_transform(split, rgb_mean, rbg_std, key='ImageNet'):

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]) if key == 'iNaturalist18' else transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std)
            ])
        }
        return data_transforms[split]

