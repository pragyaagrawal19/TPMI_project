import torch
import numpy as np
from torchvision import datasets, transforms
import random
from datasets.utils import create_loader


class MNISTLoader(object):
    def __init__(self, path, batch_size,few_sort,num_of_sample, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, few_sort,num_of_sample, transform, target_transform)
        
        # build the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset, train_sampler, batch_size,
                                          shuffle=True if train_sampler is None else False, **kwargs)

        self.test_loader = create_loader(test_dataset, test_sampler, batch_size,
                                         shuffle=False, **kwargs)

        self.output_size = 10
        #self.output_size = 1
        self.batch_size = batch_size

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])

    @staticmethod
    def get_datasets(path,few_sort,num_of_sample, transform=None, target_transform=None):
        transform_list = []
        if transform:
            assert isinstance(transform, list)
            transform_list = list(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = datasets.MNIST(path, train=True, download=True,
                                       transform=transforms.Compose(transform_list),
                                       target_transform=target_transform)
        test_dataset = datasets.MNIST(path, train=False,
                                      transform=transforms.Compose(transform_list),
                                      target_transform=target_transform)
        '''
        dataset=[train_dataset, test_dataset]
        train_ind=[]
        test_ind=[]
        Limit=num_of_sample
        index_list=[train_ind,test_ind]
       
            for data in range(0,len(dataset)):
                label_count={}
                for i in range(0,10):
                    label_count[str(i)]=0;
                
                classFull=0
                for i in range(0,len(dataset[data])):
                    _,label = dataset[data][i];
                    if(label_count[str(label)]==Limit):
                        continue;
                    else:
                        label_count[str(label)]+=1;
                        index_list[data].append(i)
                        if(label_count[str(label)]==Limit):
                            classFull+=1
                    if(classFull==10):
                        break
                        
            '''
        if few_sort:
            train_index_list=[]
            test_index_list=[]
            
            count=int(0)
            for i in range(0,num_of_sample):
                #r1=random.randrange(0,len(train_dataset))
                #r2=random.randrange(0,len(test_dataset))
                r1=random.choice(list(range(0, len(train_dataset))))
                r2=random.choice(list(range(0, len(test_dataset))))
                #print(r1,r2)
                train_index_list.append(r1)
                test_index_list.append(r2)
            #train_index_list=[1,2,3,4]
            #test_index_list=[1,2,3,4]
            #trainset_1 = torch.utils.data.Subset(train_dataset, index_list[0])
            #testset_1 = torch.utils.data.Subset(test_dataset, index_list[1])
            trainset_1 = torch.utils.data.Subset(train_dataset, train_index_list)
            testset_1 = torch.utils.data.Subset(test_dataset, test_index_list)
            print("length of train_set:",len(trainset_1))
            print("length of train_set:",len(testset_1))
            return trainset_1, testset_1
        else:
            return train_dataset, test_dataset