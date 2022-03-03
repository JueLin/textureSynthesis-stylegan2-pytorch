import os
from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset

import torch
import numpy as np
import random
import lmdb

class RandomMultiCrop(object):
    """ Obtain multiple random crops from a single input image

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
        n_crop (int): number of output crops per image
    """

    def __init__(self, output_size=256, n_crop=4, rand_90k_degree=True, rand_flip=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        assert isinstance(n_crop, int)
        self.n_crop = n_crop

        self.rand_90k_degree = rand_90k_degree
        self.rand_flip = rand_flip
    
    def __call__(self, image):
        w, h = image.size[:2]
        new_w, new_h = self.output_size

        top = np.random.randint(0, h - new_h+1, size=self.n_crop)
        left = np.random.randint(0, w - new_w+1, size=self.n_crop)

        rotate = 0 
        if self.rand_90k_degree:
            rotate = 90*np.random.randint(0, 4)
        horizontal_flip, vertical_flip = False, False
        if self.rand_flip:
            horizontal_flip = bool(random.getrandbits(1))
            vertical_flip = bool(random.getrandbits(1))
        crops = []
        for i, (t, l) in enumerate(zip(top, left)):
            crop = image.crop((l, t, l+new_w, t+new_h)).rotate(rotate)
            if horizontal_flip:
                crop = crop.transpose(method=Image.FLIP_LEFT_RIGHT)
            if vertical_flip:
                crop = crop.transpose(method=Image.FLIP_TOP_BOTTOM)
            crops.append(crop)
        return crops

class TextureDataset(Dataset):
    def __init__(self, path, transform, resolution=256):

        self.img_paths = []
        for (root, dirs, files) in os.walk(path):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    self.img_paths.append(img_path)
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img

class TextureDatasetLmdb(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path) 
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))              
                 
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).convert('RGB')
        img = self.transform(img)
        return img        
