import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
import json
from torch.utils.data import Dataset
from collections import Counter
import sys

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    

    def __init__(self, args, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True, bylabel= True):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(args.seed)
        cls_num = 10
        self.train = train
        self.bylabel = bylabel
        img_num_list = self.get_img_num_per_cls(cls_num, imb_type, imb_factor)

        self.gen_imbalanced_data(img_num_list)
        if self.bylabel == True:
            self.gen_samplebylabel_data(args)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num

        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        return img_num_per_cls



    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):

            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

        
    def get_cls_num_list(self, cls_num):
        cls_num_list = []

        for i in range(cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def gen_samplebylabel_data(self, args):
        new_data = []
        new_targets = []
        labelsamplenum = np.zeros(args.num_classes)
        totalsamplenum = 0
        labelindices = []
        for i in range(args.num_classes):
            
            labelindices.append(np.where(np.array(self.targets)==i)[0].tolist())
        weights = 1/args.num_classes*np.ones(args.num_classes)
        while totalsamplenum <(50000):
            while True:
                chosen_label = np.random.choice(range(args.num_classes), 1, p=weights)
                if labelsamplenum[chosen_label]<int(50000/args.num_classes/args.subbatch_size):
                    break
                
            tempindices = labelindices[int(chosen_label)][int(labelsamplenum[int(chosen_label)]*args.subbatch_size):
                        int((labelsamplenum[int(chosen_label)]+1)*args.subbatch_size)]
            #print(self.data[tempindices, ...].shape)
            new_data.append(self.data[tempindices, ...])
            new_targets.append([chosen_label[0], ] * args.subbatch_size)
            totalsamplenum += args.subbatch_size
            labelsamplenum[int(chosen_label)] += 1
        #print(new_data)
        #print(new_data[0].shape)
        new_data = np.array(new_data)
        #print(new_data.shape)
        self.data = new_data
        self.targets = np.array(new_targets)

        
        #print(len(self.targets))
        #sys.exit()
    """
    def __getitem__(self, index):
        #print(index)
        #print(self.data[index].shape)
        #(subbatch_size, 32, 32, 3)
        #print(self.targets[index])
        #sys.exit()
        return self.data[index], self.targets[index]
    """
    def __len__(self):
        #print(len(self.targets))
        return len(self.targets)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.bylabel==False:
            img, target = self.data[index], self.targets[index]
            #print(img.shape): (32, 32, 3)
            #print(target): int
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
            #print(img.shape):torch.size([3, 32, 32])
            if self.target_transform is not None:
                target = self.target_transform(target)
            #print(img.shape): torch.size([3, 32, 32])
            #print(target): int
            return img, target
        else:
            img, target = self.data[index], self.targets[index]
            #print(img.shape):(100, 32, 32, 3)
            #print(target): 100 list
            targetlist = []
            for i in range(img.shape[0]):
                tempimg = Image.fromarray(img[0])

                if self.transform is not None:
                    tempimg = self.transform(tempimg)
                if i==0:
                    imgnew = tempimg.unsqueeze(0)
                else:
                    imgnew = torch.vstack((imgnew, tempimg.unsqueeze(0)))

                if self.target_transform is not None:
                    temptarget = self.target_transform(target[i])
                    targetlist.append(temptarget)
                else:
                    targetlist.append(target[i])

            #print(imgnew.shape)
            #print(targetlist)

            return imgnew, targetlist



class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, args, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True, bylabel= True):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(args.seed)

        cls_num = 100
        self.train = train
        self.bylabel = bylabel
        img_num_list = self.get_img_num_per_cls(cls_num, imb_type, imb_factor)


        self.gen_imbalanced_data(img_num_list)
        if self.bylabel == True:
            self.gen_samplebylabel_data(args)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num

        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        return img_num_per_cls



    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        
        classes = np.unique(targets_np)
 
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

        
    def get_cls_num_list(self, cls_num):
        cls_num_list = []

        for i in range(cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def gen_samplebylabel_data(self, args):
        new_data = []
        new_targets = []
        labelsamplenum = np.zeros(args.num_classes)
        totalsamplenum = 0
        labelindices = []
        for i in range(args.num_classes):
            
            labelindices.append(np.where(np.array(self.targets)==i)[0].tolist())
        weights = 1/args.num_classes*np.ones(args.num_classes)
        while totalsamplenum <(50000):
            while True:
                chosen_label = np.random.choice(range(args.num_classes), 1, p=weights)
                if labelsamplenum[chosen_label]<int(50000/args.num_classes/args.subbatch_size):
                    break
                
            tempindices = labelindices[int(chosen_label)][int(labelsamplenum[int(chosen_label)]*args.subbatch_size):
                        int((labelsamplenum[int(chosen_label)]+1)*args.subbatch_size)]
            #print(self.data[tempindices, ...].shape)
            new_data.append(self.data[tempindices, ...])
            new_targets.append([chosen_label[0], ] * args.subbatch_size)
            totalsamplenum += args.subbatch_size
            labelsamplenum[int(chosen_label)] += 1
        #print(new_data)
        #print(new_data[0].shape)
        new_data = np.array(new_data)
        #print(new_data.shape)
        self.data = new_data
        self.targets = np.array(new_targets)

        
        #print(len(self.targets))
        #sys.exit()
    """
    def __getitem__(self, index):
        #print(index)
        #print(self.data[index].shape)
        #(subbatch_size, 32, 32, 3)
        #print(self.targets[index])
        #sys.exit()
        return self.data[index], self.targets[index]
    """
    def __len__(self):
        #print(len(self.targets))
        return len(self.targets)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.bylabel==False:
            img, target = self.data[index], self.targets[index]
            #print(img.shape): (32, 32, 3)
            #print(target): int
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
            #print(img.shape):torch.size([3, 32, 32])
            if self.target_transform is not None:
                target = self.target_transform(target)
            #print(img.shape): torch.size([3, 32, 32])
            #print(target): int
            return img, target
        else:
            img, target = self.data[index], self.targets[index]
            #print(img.shape):(100, 32, 32, 3)
            #print(target): 100 list
            targetlist = []
            for i in range(img.shape[0]):
                tempimg = Image.fromarray(img[0])

                if self.transform is not None:
                    tempimg = self.transform(tempimg)
                if i==0:
                    imgnew = tempimg.unsqueeze(0)
                else:
                    imgnew = torch.vstack((imgnew, tempimg.unsqueeze(0)))

                if self.target_transform is not None:
                    temptarget = self.target_transform(target[i])
                    targetlist.append(temptarget)
                else:
                    targetlist.append(target[i])

            #print(imgnew.shape)
            #print(targetlist)

            return imgnew, targetlist

    # def _check_integrity(self):
    #     return True



def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic



def get_num_classes(args):
    num_classes = 0
    if args.dataset == 'ina':
        num_classes = 1010
    elif args.dataset == 'imagenet-LT':
        num_classes = 1000
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    elif args.dataset == 'places-LT':
        num_classes = 365
    elif args.dataset == 'covid-LT':
        num_classes = 4
    elif args.dataset == 'iNaturalist18':
        num_classes = 8142
    return num_classes

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()

