import numpy as np
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
import random
from lib import get_mask
from utils.data.adain import vgg, decoder, style_transfer


import matplotlib.pyplot as plt


vgg_path = './models/vgg_normalised.pth'
decoder_path = './models/decoder.pth'
img_size = 320

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn)
    ]
    return albu.Compose(_transform)


class CustomDataset(Dataset):
    def __init__(self, data_path, masks=True, preprocessing=None, transforms=None, mixup_proba=None, style_transfer_path=None, device='cuda'):
        if style_transfer_path is not None:
            self.device = device
            self.vgg = vgg
            self.decoder = decoder
            self.decoder.eval()
            self.vgg.eval()
            self.vgg.load_state_dict(torch.load(vgg_path, map_location=self.device))
            self.vgg = torch.nn.Sequential(*list(self.vgg.children())[:31])
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            self.decoder.to(self.device)
            self.vgg.to(self.device)
            self.style_files = os.listdir(style_transfer_path)
        
        
        self.style_path = style_transfer_path

        self.files = os.listdir(data_path)
        
        self.path = data_path    
        self.masks = masks
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.mixup_proba = mixup_proba

    def __getitem__(self, idx):
        if self.masks:
            mask = cv2.imread(f"{self.path}_mask/{self.files[idx].split('.')[0] + '.png'}")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        img = cv2.imread(f"{self.path}/{self.files[idx]}")
        
        if self.mixup_proba is not None:
            if random.random() <= self.mixup_proba:
                idx2 = random.randint(0, len(self.files)-1)
                if self.masks:
                    mask2 = cv2.imread(f"{self.path}_mask/{self.files[idx2].split('.')[0] + '.png'}")
                    mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
                    mask = (0.5 * mask2 + 0.5 * mask).astype('uint8')
                    mask[mask > 0] = 255
                img2 = cv2.imread(f"{self.path}/{self.files[idx2]}")
                img = (0.5 * img2 + 0.5 * img).astype('uint8')
                            
        img = cv2.resize(img, (img_size, img_size))
        if self.masks:
            mask = cv2.resize(mask, (img_size, img_size))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.style_path is not None:
            style_sample_idx = random.randint(0, len(self.style_files) - 1)
            style_sample = self.style_files[style_sample_idx]
            style_img = cv2.imread(f"{self.style_path}/{style_sample}")
            style_img = cv2.resize(style_img, (img_size, img_size))
            style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
            
            img = style_transfer(self.vgg, self.decoder, 
                                 torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(self.device), 
                                 torch.from_numpy(np.transpose(style_img, (2, 0, 1))).unsqueeze(0).float().to(self.device), 
                                 alpha=0.15, interpolation_weights=None)
        
            img = img.squeeze(0).detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype('uint8')
        
        if self.masks:
            if self.transforms is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                
            if self.preprocessing is not None:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed['image']
        else:
            if self.preprocessing is not None:
                preprocessed = self.preprocessing(image=img)
                img = preprocessed['image']

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        
        if self.masks:
            mask = torch.from_numpy(mask)        
            mask = (mask > 0).float()
            return img, mask
        else:
            return img

    def __len__(self):
        return len(self.files)