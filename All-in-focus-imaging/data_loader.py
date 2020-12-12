import h5py
import torch.utils.data as data
import numpy as np
import torch
import os
import os.path
import skimage.transform
from skimage.io import imread
import torch.utils.data
import torchvision
from torchvision import transforms
import re
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler
import random
import numbers
from PIL import Image
import collections
import sys
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

def make_dataset(list_name, dpt_list_name):
    text_file = open(list_name, 'r')
    images_list = text_file.readlines()
    text_file.close()
    images_list = [os.path.join(os.getcwd(), i) for i in images_list]
    #print(images_list)
    
    text_file = open(dpt_list_name, 'r')
    dpt_list = text_file.readlines()
    text_file.close()
    dpt_list = [os.path.join(os.getcwd(), i) for i in dpt_list]
    #print(dpt_list)
    
    return images_list, dpt_list

def _get_image_size(img):
    if TF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class SeqToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic_seq, dpt_seq):
        for i, (pic, dpt) in enumerate(zip(pic_seq, dpt_seq)):
            pic_seq[i] = TF.to_pil_image(pic, self.mode)
            dpt_seq[i] = TF.to_pil_image(dpt, self.mode)
        return pic_seq, dpt_seq
    
class SeqRandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img_seq, dpt_seq):
        randnum = random.random()
        for i, (img, dpt) in enumerate(zip(img_seq, dpt_seq)):
            if randnum < self.p:
                img_seq[i] = TF.hflip(img)
                dpt_seq[i] = TF.hflip(dpt)
        return img_seq, dpt_seq
    
class SeqRandomVerticalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img_seq, dpt_seq):
        randnum = random.random()
        for i, (img, dpt) in enumerate(zip(img_seq, dpt_seq)):
            if randnum < self.p:
                img_seq[i] = TF.vflip(img)
                dpt_seq[i] = TF.vflip(dpt)
        return img_seq, dpt_seq
    
class SeqRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_seq, dpt_seq):

        i, j, h, w = self.get_params(dpt_seq[0], self.size)

        for i, (img, dpt) in enumerate(zip(img_seq, dpt_seq)):
            img_seq[i] = TF.crop(img, i, j, h, w)
            dpt_seq[i] = TF.crop(dpt, i, j, h, w)
        return img_seq, dpt_seq
    
class SeqResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_seq, dpt_seq):
        for i, (img, dpt) in enumerate(zip(img_seq, dpt_seq)):
            img_seq[i] = TF.resize(img, self.size, self.interpolation)
            dpt_seq[i] = TF.resize(dpt, self.size, self.interpolation)
        return img_seq, dpt_seq
    
class SeqToTensor(object):
    def __call__(self, img_seq, dpt_seq):
        for i, (img, dpt) in enumerate(zip(img_seq, dpt_seq)):
            img_seq[i] = TF.to_tensor(img)
            dpt_seq[i] = TF.to_tensor(dpt)
        return torch.stack(img_seq), torch.stack(dpt_seq)
    
class ComposedTransforms(object):
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_seq, dpt_seq):
        for t in self.transforms:
            img_seq, dpt_seq = t(img_seq, dpt_seq)
        return img_seq, dpt_seq          
    
class DAVISImageFolder(data.Dataset):

    def __init__(self, list_path, dpt_list_path, seq, transform = ComposedTransforms([
                                SeqToPILImage(),
                                SeqResize((1536, 3072)),
                                SeqToTensor()])):
        img_list, dpt_list = make_dataset(list_path, dpt_list_path)
        if len(img_list) == 0:
            raise RuntimeError('Found 0 images in: ' + list_path)
        self.img_list = img_list
        self.dpt_list = dpt_list

        self.seq = seq
        self.transform = transform
        self.offset = 2
        
    def load_imgs(self, img_path):
        img = imread(img_path)
        return img
    
    def load_dpt(self, dpt_path):
        dpt = (np.load(dpt_path) - 1.0)/4.0*6.0 + 1.0
        dpt = np.float32(skimage.transform.resize(dpt, (480, 854)))

        return dpt

    def __getitem__(self, index):
        
        imgSeq = []
        dptSeq = []
        
        for i in range(self.seq):
            h5_path = self.img_list[index].rstrip()
            dpt_h5_path = self.dpt_list[index].rstrip()
            
            try:
                h5_path_offset = self.replace_num_in_string(h5_path, i*self.offset)
                dpt_h5_path_offset = self.replace_num_in_string(dpt_h5_path, i*self.offset)
                img = self.load_imgs(h5_path_offset)
                dpt = self.load_dpt(dpt_h5_path_offset)
            except FileNotFoundError:
                img = self.load_imgs(h5_path)
                dpt = self.load_dpt(dpt_h5_path)
            imgSeq.append(img)
            dptSeq.append(dpt)

        imgSeq, dptSeq = self.transform(imgSeq, dptSeq)
        imgSeq = torch.clamp(imgSeq*2.0-1.0, -1.0, 1.0)
        dptSeq = torch.clamp(dptSeq, 1.0, 7.0)
        
        return imgSeq, dptSeq

    def __len__(self):
        return len(self.img_list)

    def replace_num_in_string(self, string, offset):
        if offset == 0:
            return string
        regex = r"00[0-9][0-9][0-9]"
        matches = re.search(regex, string)
        try:
            matches.group()
        except:
            print(string)
        subst = "{:05d}".format(int(matches.group()) + offset)
        result = re.sub(regex, subst, string, 0, re.MULTILINE)
        return result
    
    
class DAVISDataLoader():
    def __init__(self, name, list_path, dpt_list_path, seq, _batch_size, valid_size = 0.2, shuffle = True):
        if name == 'sintel':
            dataset = SintelImageFolder(list_path=list_path, dpt_list_path=dpt_list_path, seq = seq)
        else:
            dataset = DAVISImageFolder(list_path=list_path, dpt_list_path=dpt_list_path, seq = seq)
            
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        if shuffle:
            np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
            
        self.train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False,
                                                       num_workers=int(8))
        self.valid_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       sampler=valid_sampler,
                                                       shuffle=False,
                                                       num_workers=int(8))
        self.dataset = dataset

    def load_data(self):
        return (self.train_loader, self.valid_loader)

    def __len__(self):

        return len(self.dataset)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(20,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_davis_dataset(video_path, depth_path, seq=3, batch_size = 1):
    video_list = video_path
    dpt_list = depth_path
    video_data_loader = DAVISDataLoader('davis', video_list, dpt_list, seq, batch_size)
    video_dataset = video_data_loader.load_data()
    return video_dataset

