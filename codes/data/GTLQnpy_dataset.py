import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import sys
import os
import time
# try:
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from data.util import imresize_np
#     from utils import util as utils
# except ImportError:
#     pass

# import random
# import numpy as np
# import cv2
# import lmdb
# import torch
# import torch.utils.data as data
# import data.util as util


class GTLQnpyDataset(data.Dataset):
    '''
    Load  HR-LR image npy pairs. Make sure HR-LR images are in the same order.
    '''

    def __init__(self, opt):
        super(GTLQnpyDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT, self.paths_Noisy = None, None, None
        self.sizes_LQ, self.sizes_GT, self.sizes_Noisy = None, None, None
        self.LQ_env, self.GT_env, self.Noisy_env = None, None, None  # environment for lmdb
        self.scale = opt['scale']
        if self.opt['phase'] == 'train':
            self.GT_size = opt['GT_size']
            self.LR_size = self.GT_size // self.scale

        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])  # LR list
        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])  # GT list
        self.paths_Noisy, self.sizes_Noisy = util.get_image_paths(self.data_type, opt['dataroot_Noisy'])

        assert self.paths_GT, 'Error: GT paths are empty.'
        assert self.paths_Noisy, 'Error: Noisy path is empty.'

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT), 'GT and LR datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):
        GT_path, Noisy_path, LR_path = None, None, None
        # get GT and LR image
        GT_path = self.paths_GT[index]
        # LR_path = self.paths_LQ[index]
        Noisy_path = self.paths_Noisy[index]
        # LR_path = GT_path.replace('DIV2K+Flickr2K_HR', 'DIV2K+Flickr2K_LR_bicubic/X4').replace('.npy','x{}.npy'.format(self.scale))
        time_start = time.time()
        img_GT = util.read_img_fromnpy(np.load(GT_path))
        time_end = time.time()
        time_spent = time_end - time_start
        print('GT img time' + str(time_spent))

        time_start = time.time()
        img_Noisy = util.read_img_fromnpy(np.load(Noisy_path))
        time_end = time.time()
        time_spent = time_end - time_start
        print('Noisy img time' + str(time_spent))

        time_start = time.time()
        if self.paths_LQ:
            LR_path = self.paths_LQ[index]
            img_LR = util.read_img_fromnpy(np.load(LR_path))  # return: Numpy float32, HWC, BGR, [0,1]
        else:
            img_LR = util.imresize_np(img_GT, 1 / self.scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)
        time_end = time.time()
        time_spent = time_end - time_start
        print('LQ img time' + str(time_spent))

        
        
        if self.opt['phase'] == 'train':
            # crop
            time_start = time.time()
            H, W, C = img_LR.shape
            rnd_top_LR = random.randint(0, max(0, H - self.LR_size))
            rnd_left_LR = random.randint(0, max(0, W - self.LR_size))
            rnd_top_GT = rnd_top_LR * self.scale
            rnd_left_GT = rnd_left_LR * self.scale

            img_GT = img_GT[rnd_top_GT:rnd_top_GT + self.GT_size, rnd_left_GT:rnd_left_GT + self.GT_size, :]
            img_Noisy = img_Noisy[rnd_top_GT:rnd_top_GT + self.GT_size, rnd_left_GT:rnd_left_GT + self.GT_size, :]
            img_LR = img_LR[rnd_top_LR:rnd_top_LR + self.LR_size, rnd_left_LR:rnd_left_LR + self.LR_size, :]
            time_end = time.time()
            time_spent = time_end - time_start
            print('crop img time' + str(time_spent))

            # augmentation - flip, rotate
            time_start = time.time()
            img_LR, img_GT, img_Noisy = util.augment([img_LR, img_GT, img_Noisy], self.opt['use_flip'],
                                          self.opt['use_rot'])
            time_end = time.time()
            time_spent = time_end - time_start
            print('augmentation time' + str(time_spent))
            # img_GT, img_LR = util.augment([img_GT, img_LR], self.opt['use_flip'],
                                #   self.opt['use_rot'], self.opt['mode'])

        # change color space if necessary, deal with gray image
        time_start = time.time()
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_Noisy = util.channel_convert(img_Noisy.shape[2], self.opt['color'], [img_Noisy])[0]
            img_LR = util.channel_convert(img_LR.shape[2], self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if img_Noisy.shape[2] == 3:
            img_Noisy = img_Noisy[:, :, [2, 1, 0]]
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_Noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Noisy, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        time_end = time.time()
        time_spent = time_end - time_start
        print('color space time' + str(time_spent))
        if LR_path is None:
            LR_path = GT_path

        return {'LQ': img_LR, 'Noisy':img_Noisy, 'GT': img_GT, 'LQ_path': LR_path, 'GT_path': GT_path}


    def __len__(self):
        return len(self.paths_GT)

