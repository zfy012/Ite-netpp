import os
import numpy as np

import nibabel as nib
import torch
from torch.utils.data import Dataset
from augmentation import combine_augment


class ToTensor(object):
    def __call__(self, patch):
        if patch.dtype != np.float32:
            patch = patch.astype('float32')

        return torch.from_numpy(patch)


class Normalize(object):
    def __call__(self, patch):
        patch = patch.astype('float32')

        mean = patch.mean()
        std = patch.std()
        if std > 0:
            ret = (patch - mean) / std
        else:
            ret = patch * 0.

        return ret


class ImagePool(Dataset):
    def __init__(self, img_dir, seg_dir, transform=None, mode='train'):
        super(ImagePool, self).__init__()

        self.img_list = list(
            map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))

        self.seg_list = list(
            map(lambda x: os.path.join(seg_dir, x), os.listdir(seg_dir)))

        assert len(self.img_list) == len(self.seg_list)

        self.img_list = sorted(self.img_list)
        self.seg_list = sorted(self.seg_list)

        self.transform = transform

        assert mode in ['train', 'valid', 'test']
        self.mode = mode

    def __getitem__(self, index):
        img_file = self.img_list[index]
        seg_file = self.seg_list[index]

        input_img = nib.load(img_file)
        img_data = input_img.get_fdata()
        input_seg = nib.load(seg_file)
        seg_data = input_seg.get_fdata()

        if self.mode == 'train':
            # data augmentation
            img_data, seg_data = combine_augment(img_data, seg_data)

        img_data = np.expand_dims(img_data, axis=0).astype('float32')   #[C, D, H, W]
        seg_data = np.expand_dims(seg_data, axis=0).astype('float32')   #[C, D, H, W]

        if self.transform is not None:
            img_data = self.transform(img_data)

        return {'img': img_data, 'seg': seg_data, 'name': seg_file}

    def __len__(self):
        return len(self.img_list)
