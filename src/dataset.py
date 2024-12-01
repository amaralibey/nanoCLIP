# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pathlib
from collections import defaultdict
from PIL import Image
import random


class Flickr30k(Dataset):
    def __init__(self, base_path, split='train', img_transform=None, txt_transform=None):
        base_path = pathlib.Path(base_path)
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        
        self.img_dir = base_path / 'flickr30k_images'
        self.split = split
        
        # load all captions
        self.captions = defaultdict(list)
        with open(base_path / 'captions.txt', 'r') as f:
            for line in f.readlines()[1:]: # ignore the header (first line)
                image, caption_number, caption = line.strip().split(',', 2)
                self.captions[image].append(caption)
        
        # get all image names
        self.imgs = list(self.captions.keys())

        # split the dataset
        if split == 'train':
            self.imgs = self.imgs[ : int(0.8 * len(self.imgs))]
        elif split == 'val':
            self.imgs = self.imgs[int(0.8 * len(self.imgs)) : ]
        else: # use all images
            pass
            
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(self.img_dir / img_name).convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)

        captions = self.captions[img_name]
        if self.txt_transform:
            captions = [self.txt_transform(caption) for caption in captions]
        return img, captions
    

class CollateFlickr:
    """    
        Collate class for the dataloader (to be called in the dataloader)
        This will be called for each batch of data
        It will convert the list of images and captions into a single tensor
        The captions will be tokenized and padded to the max_length 
        The images will be stacked into a single tensor
    """
    def __init__(self, tokenizer, max_length=80, captions_to_use='all'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.captions_to_use = captions_to_use
        
    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        
        if self.captions_to_use == 'first':
            captions = [caption[0] for caption in captions]
        elif self.captions_to_use == 'random':
            captions = [caption[random.randint(0, 4)] for caption in captions]
        elif self.captions_to_use == 'all':
            pass # use all captions
        else:
            raise ValueError("captions_to_use should be one of 'all', 'first', 'random'")
        
        # captions are either a list of strings or a list of list of strings
        captions_ids  = []
        masks = []
        if isinstance(captions[0], list): # list of list of strings               
            # multiple captions
            for caption_list in captions:
                caps = [self.tokenizer(caption, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt") for caption in caption_list]
                captions_ids.append(torch.stack([caption['input_ids'].squeeze(0) for caption in caps]))
                masks.append(torch.stack([caption['attention_mask'].squeeze(0) for caption in caps]))
            
            captions_ids = torch.stack(captions_ids)
            masks = torch.stack(masks)        
        else:
            # single caption
            captions = self.tokenizer(captions, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
            captions_ids = captions['input_ids'].squeeze(0)
            masks = captions['attention_mask'].squeeze(0)
        
        return images, captions_ids, masks