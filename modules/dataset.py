import numpy as np
import os

import time
import glob
import math

from PIL import Image, ImageFilter
import random 

from scipy import optimize

import torch
from torch.utils.data import Dataset

import cv2

from tqdm import tqdm

import copy
from skimage.segmentation import flood, flood_fill
import imgaug.augmenters as iaa

from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage

def load_image(file):
    return Image.fromarray(np.array(Image.open(file).convert('RGB'))[:, :, 0], 'L')
    
class AllSegmDataset():
    def __init__(self, image_dir_coarse, mask_dir_coarse, image_dir_openeds, mask_dir_openeds, augment=True, input_transform=None, target_transform=None, heavy_augment_prob = 0.3):
        self.image_dir_coarse = image_dir_coarse
        self.mask_dir_coarse = mask_dir_coarse
        
        self.image_dir_openeds = image_dir_openeds
        self.mask_dir_openeds = mask_dir_openeds
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        
        self.data = []
        
        if self.image_dir_coarse[-1] == '/':
            all_image_paths_coarse = glob.glob(self.image_dir_coarse + '*.*')
        else:
            all_image_paths_coarse = glob.glob(self.image_dir_coarse + '/*.*')
        
        if self.image_dir_openeds[-1] == '/':
            all_image_paths_openeds = glob.glob(self.image_dir_openeds + '*.*')
        else:
            all_image_paths_openeds = glob.glob(self.image_dir_openeds + '/*.*')
            
        for image_path in all_image_paths_coarse:
            self.data.append([image_path, "coarse"])
        
        for image_path in all_image_paths_openeds:
            self.data.append([image_path, "openeds"])
        
        self.length = len(self.data)
        self.heavy_augment_prob = heavy_augment_prob
    
    def set_augment(self, val):
        print('Augmenting?', val)
        self.augment = val
    
    def set_heavy_augment_prob(self, prob):
        self.heavy_augment_prob = prob
        print('Heavy Augmentation Probability set to:', self.heavy_augment_prob)
        
    def __getitem__(self, index):
        image = load_image(self.data[index][0])
        dataset_tag = self.data[index][1]
        
        w, h = image.size
        
        new_w = 320
        new_h = 240
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        image_name = self.data[index][0].split('/')[-1]
        
        if dataset_tag == "coarse":
            mask = load_image(os.path.join(self.mask_dir_coarse, image_name))
            mask = mask.resize((new_w, new_h), Image.NEAREST)
            mask = np.array(mask)
            mask = np.where(mask > 0, 2, 0).astype(np.intc)
        elif dataset_tag == "openeds":
            mask = np.load(os.path.join(self.mask_dir_openeds, image_name.split('.')[0] + '.npy'))
            mask = mask.astype(np.intc)
            mask = cv2.resize(mask, (new_w, new_h), interpolation = cv2.INTER_NEAREST_EXACT)
        
        if self.augment:
            # horizontal flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = np.flip(mask, axis=1)
            
            # affine transformations
            if random.random() < 0.8:
                aug = iaa.Affine(scale=(0.7, 1.4), translate_px={"x": (-40, 40), "y": (-40, 40)}, rotate=(-45, 45), mode=random.choice(['constant', 'edge']), cval=127)
                img_np, mask_np = aug(images = np.expand_dims(np.expand_dims(image, axis=0), axis=-1), segmentation_maps = np.expand_dims(np.expand_dims(mask, axis=0), axis=-1))
                image = Image.fromarray(np.uint8(img_np[0, :, :, 0]))
                mask = mask_np[0, :, :, 0]
                
            if random.random() < self.heavy_augment_prob:
                if random.random() < 0.5:
                    random_choice = np.random.choice([1,2,3,4,5])
                    if random_choice == 1:   
                        # sharpening
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.DETAIL)
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.SHARPEN)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.EDGE_ENHANCE)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                        else:
                            aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 2:
                        #blurring
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.GaussianBlur())
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.BLUR)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.SMOOTH)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.SMOOTH_MORE)
                        else:
                            aug = iaa.MotionBlur(k=3)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 3:
                        #noise
                        if random.random() < 0.5:
                            aug = iaa.AdditiveGaussianNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                        else:
                            aug = iaa.AdditiveLaplaceNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 4:
                        # Basic compression and expansion
                        divider = random.random() + 1.1
                        cw, ch = image.size
                        new_cw = int(cw/divider)
                        new_ch = int(ch/divider)
                        
                        first_choice = np.random.choice([1,2,3,4,5,6])
                        if first_choice == 1:
                            image = image.resize((new_cw, new_ch), Image.NEAREST)
                        elif first_choice == 2:
                            image = image.resize((new_cw, new_ch), Image.BILINEAR)
                        elif first_choice == 3:
                            image = image.resize((new_cw, new_ch), Image.BICUBIC)
                        elif first_choice == 4:
                            image = image.resize((new_cw, new_ch), Image.LANCZOS)
                        elif first_choice == 5:
                            image = image.resize((new_cw, new_ch), Image.HAMMING)
                        else:
                            image = image.resize((new_cw, new_ch), Image.BOX)
                        
                        second_choice = np.random.choice([1,2,3,4,5,6])
                        if second_choice == 1:
                            image = image.resize((cw, ch), Image.NEAREST)
                        elif second_choice == 2:
                            image = image.resize((cw, ch), Image.BILINEAR)
                        elif second_choice == 3:
                            image = image.resize((cw, ch), Image.BICUBIC)
                        elif second_choice == 4:
                            image = image.resize((cw, ch), Image.LANCZOS)
                        elif second_choice == 5:
                            image = image.resize((cw, ch), Image.HAMMING)
                        else:
                            image = image.resize((cw, ch), Image.BOX)
                    else:
                        #JPEG compression
                        aug = iaa.JpegCompression(compression=(50, 80))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                
                if random.random() < 0.5: # random contrast change
                    random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                    if random_degree == 1:
                        aug = iaa.GammaContrast((0.5, 2.0))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 2:
                        aug = iaa.LinearContrast((0.4, 1.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 3:
                        aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 4:
                        aug = iaa.LogContrast(gain=(0.6, 1.4))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])            
                    else:    
                        aug = iaa.pillike.EnhanceBrightness()
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])        
        
        if dataset_tag == "coarse":
            mask_for_loss = torch.from_numpy(np.where(mask > 0, True, False)).bool()
        elif dataset_tag == "openeds":
            mask_for_loss = torch.from_numpy(np.ones(mask.shape)).bool()
        
        if self.input_transform is not None:
            image = self.input_transform(image)
        
        if self.target_transform is not None:
            mask_t = self.target_transform(mask)
        else:
            mask_t = torch.from_numpy(np.copy(mask))
        
        data = {'image' : image, 'mask' : mask_t, 'mask_for_loss' :mask_for_loss, "dataset" : dataset_tag}
        
        return data
        
    def __len__(self):
        return self.length
    
class ScleraSegmDataset():
    def __init__(self, image_dir, mask_dir, augment=True, input_transform=None, target_transform=None, heavy_augment_prob = 0.3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        
        self.data = []
        
        if self.image_dir[-1] == '/':
            self.data = glob.glob(self.image_dir + '*.*')
        else:
            self.data = glob.glob(self.image_dir + '/*.*')
        
        self.length = len(self.data)
        self.heavy_augment_prob = heavy_augment_prob
    
    def set_augment(self, val):
        print('Augmenting?', val)
        self.augment = val
    
    def set_heavy_augment_prob(self, prob):
        self.heavy_augment_prob = prob
        print('Heavy Augmentation Probability set to:', self.heavy_augment_prob)
        
    def __getitem__(self, index):
        image = load_image(self.data[index])
        w, h = image.size
        
        new_w = 320
        new_h = 240
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        image_name = self.data[index].split('/')[-1]
        
        mask = np.load(os.path.join(self.mask_dir, image_name.split('.')[0] + '.npy'))
        mask = mask.astype(np.intc)
        mask = cv2.resize(mask, (new_w, new_h), interpolation = cv2.INTER_NEAREST_EXACT)
        
        if self.augment:
            # horizontal flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = np.flip(mask, axis=1)
            
            # affine transformations
            if random.random() < 0.8:
                aug = iaa.Affine(scale=(0.7, 1.4), translate_px={"x": (-40, 40), "y": (-40, 40)}, rotate=(-45, 45), mode=random.choice(['constant', 'edge']), cval=127)
                img_np, mask_np = aug(images = np.expand_dims(np.expand_dims(image, axis=0), axis=-1), segmentation_maps = np.expand_dims(np.expand_dims(mask, axis=0), axis=-1))
                image = Image.fromarray(np.uint8(img_np[0, :, :, 0]))
                mask = mask_np[0, :, :, 0]
                
            if random.random() < self.heavy_augment_prob:
                if random.random() < 0.5:
                    random_choice = np.random.choice([1,2,3,4,5])
                    if random_choice == 1:   
                        # sharpening
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.DETAIL)
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.SHARPEN)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.EDGE_ENHANCE)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                        else:
                            aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 2:
                        #blurring
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.GaussianBlur())
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.BLUR)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.SMOOTH)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.SMOOTH_MORE)
                        else:
                            aug = iaa.MotionBlur(k=3)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 3:
                        #noise
                        if random.random() < 0.5:
                            aug = iaa.AdditiveGaussianNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                        else:
                            aug = iaa.AdditiveLaplaceNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 4:
                        # Basic compression and expansion
                        divider = random.random() + 1.1
                        cw, ch = image.size
                        new_cw = int(cw/divider)
                        new_ch = int(ch/divider)
                        
                        first_choice = np.random.choice([1,2,3,4,5,6])
                        if first_choice == 1:
                            image = image.resize((new_cw, new_ch), Image.NEAREST)
                        elif first_choice == 2:
                            image = image.resize((new_cw, new_ch), Image.BILINEAR)
                        elif first_choice == 3:
                            image = image.resize((new_cw, new_ch), Image.BICUBIC)
                        elif first_choice == 4:
                            image = image.resize((new_cw, new_ch), Image.LANCZOS)
                        elif first_choice == 5:
                            image = image.resize((new_cw, new_ch), Image.HAMMING)
                        else:
                            image = image.resize((new_cw, new_ch), Image.BOX)
                        
                        second_choice = np.random.choice([1,2,3,4,5,6])
                        if second_choice == 1:
                            image = image.resize((cw, ch), Image.NEAREST)
                        elif second_choice == 2:
                            image = image.resize((cw, ch), Image.BILINEAR)
                        elif second_choice == 3:
                            image = image.resize((cw, ch), Image.BICUBIC)
                        elif second_choice == 4:
                            image = image.resize((cw, ch), Image.LANCZOS)
                        elif second_choice == 5:
                            image = image.resize((cw, ch), Image.HAMMING)
                        else:
                            image = image.resize((cw, ch), Image.BOX)
                    else:
                        #JPEG compression
                        aug = iaa.JpegCompression(compression=(50, 80))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                
                if random.random() < 0.5: # random contrast change
                    random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                    if random_degree == 1:
                        aug = iaa.GammaContrast((0.5, 2.0))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 2:
                        aug = iaa.LinearContrast((0.4, 1.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 3:
                        aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 4:
                        aug = iaa.LogContrast(gain=(0.6, 1.4))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])            
                    else:    
                        aug = iaa.pillike.EnhanceBrightness()
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])        
        
        if self.input_transform is not None:
            image = self.input_transform(image)
        
        if self.target_transform is not None:
            mask_t = self.target_transform(mask)
        else:
            mask_t = torch.from_numpy(np.copy(mask))
        
        data = {'image' : image, 'mask' : mask_t}
        
        return data
        
    def __len__(self):
        return self.length
    
class IrisSegmDataset():
    def __init__(self, image_dir, mask_dir, augment=False, input_transform=None, target_transform=None, heavy_augment_prob=0.3):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        
        if self.image_dir[-1] == '/':
            all_image_paths = glob.glob(self.image_dir + '*.*')
        else:
            all_image_paths = glob.glob(self.image_dir + '/*.*')
            
        self.image_paths = []
        self.mask_paths = []
        print(len(all_image_paths))
        
        for image_path in all_image_paths:
            image_name = image_path.split('/')[-1]
            mask_path = os.path.join(self.mask_dir, image_name)
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)
            
        self.length = len(self.image_paths)
        self.heavy_augment_prob = heavy_augment_prob
        
    def set_augment(self, val):
        print('Augmenting?', val)
        self.augment = val
    
    def set_augment_prob(self, prob):
        self.heavy_augment_prob = prob
        print('Heavy Augmentation Probability set to:', self.heavy_augment_prob)

    def __getitem__(self, index):
        # Get the image and the mask
        image = load_image(self.image_paths[index])
        mask = load_image(self.mask_paths[index])

        # Resize image and mask
        image = image.resize((320, 240), Image.BILINEAR)
        mask = mask.resize((320, 240), Image.NEAREST)

        # Data augmentation
        if self.augment:
            # horizontal flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT) 
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Affine Transformations
            if random.random() < 0.8:
                aug = iaa.Affine(scale=(0.7, 1.4), translate_px={"x": (-40, 40), "y": (-40, 40)}, rotate=(-45, 45), mode=random.choice(['constant', 'edge']), cval=127)
                img_np, mask_np = aug(images = np.expand_dims(np.expand_dims(np.uint8(image), axis=0), axis=-1), segmentation_maps = np.expand_dims(np.expand_dims(np.uint8(mask), axis=0), axis=-1))
                image = Image.fromarray(np.uint8(img_np[0, :, :, 0]))
                mask_np = np.where(mask_np>127, 255, 0)
                mask = Image.fromarray(np.uint8(mask_np[0, :, :, 0]))
            
            
                
            if random.random() < self.heavy_augment_prob:
                if random.random() < 0.5:
                    random_choice = np.random.choice([1,2,3])
                    if random_choice == 1:   
                        # sharpening
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.DETAIL)
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.SHARPEN)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.EDGE_ENHANCE)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                        else:
                            aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 2:
                        #blurring
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.GaussianBlur())
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.BLUR)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.SMOOTH)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.SMOOTH_MORE)
                        else:
                            aug = iaa.MotionBlur(k=3)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    else:
                        if random.random() < 0.5:
                            aug = iaa.AdditiveGaussianNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                        else:
                            aug = iaa.AdditiveLaplaceNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                if random.random() < 0.5:
                    if random.random() < 0.5:
                        # Basic compression and expansion
                        divider = random.random() + 1.1
                        cw, ch = image.size
                        new_cw = int(cw/divider)
                        new_ch = int(ch/divider)
                        
                        first_choice = np.random.choice([1,2,3,4,5,6])
                        if first_choice == 1:
                            image = image.resize((new_cw, new_ch), Image.NEAREST)
                        elif first_choice == 2:
                            image = image.resize((new_cw, new_ch), Image.BILINEAR)
                        elif first_choice == 3:
                            image = image.resize((new_cw, new_ch), Image.BICUBIC)
                        elif first_choice == 4:
                            image = image.resize((new_cw, new_ch), Image.LANCZOS)
                        elif first_choice == 5:
                            image = image.resize((new_cw, new_ch), Image.HAMMING)
                        else:
                            image = image.resize((new_cw, new_ch), Image.BOX)
                        
                        second_choice = np.random.choice([1,2,3,4,5,6])
                        if second_choice == 1:
                            image = image.resize((cw, ch), Image.NEAREST)
                        elif second_choice == 2:
                            image = image.resize((cw, ch), Image.BILINEAR)
                        elif second_choice == 3:
                            image = image.resize((cw, ch), Image.BICUBIC)
                        elif second_choice == 4:
                            image = image.resize((cw, ch), Image.LANCZOS)
                        elif second_choice == 5:
                            image = image.resize((cw, ch), Image.HAMMING)
                        else:
                            image = image.resize((cw, ch), Image.BOX)
                    else:
                        #JPEG compression
                        aug = iaa.JpegCompression(compression=(10, 30))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                             
                if random.random() < 0.5: # random contrast change
                    random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                    if random_degree == 1:
                        aug = iaa.GammaContrast((0.5, 2.0))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 2:
                        aug = iaa.LinearContrast((0.4, 1.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 3:
                        aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 4:
                        aug = iaa.LogContrast(gain=(0.6, 1.4))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])            
                    else:    
                        aug = iaa.pillike.EnhanceBrightness()
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    
        if self.input_transform is not None:
            image = self.input_transform(image)
        
        if self.target_transform is not None:
            mask = np.array(mask) 
            mask = self.target_transform(mask)
        else:
            mask = ToTensor()(mask)
        
        data = {'image': image, 'mask': mask}

        return data

    def __len__(self):
        return self.length
        
class IrisSegmDatasetCircle():
    def __init__(self, circle_param_csv='/afs/crc.nd.edu/user/a/avanave2/OpenSourceIrisRecognition/methods/HDBIF/Matlab/BXGRID-Iris-Full-results.csv', augment=False, input_transform=None, train_on_image=True):
        super().__init__()
        self.circle_param_csv = circle_param_csv
        
        self.input_transform = input_transform
        self.augment = augment
            
        self.data = []
        
        with open(self.circle_param_csv, 'r') as circleCSVFile:
            #skip the header
            circleCSVFile.readline()

            for line in circleCSVFile:
                lineparts = line.strip().split(',')
                self.data.append((lineparts[0], float(lineparts[1]), float(lineparts[2]), float(lineparts[3]), float(lineparts[4]), float(lineparts[5]), float(lineparts[6])))
                
        self.length = len(self.data)
        
        self.heavy_augment_prob = 0.3
        
        self.train_on_image = train_on_image
        
    def set_augment(self, val):
        print('Augmenting?', val)
        self.augment = val
    
    def set_augment_prob(self, prob):
        self.heavy_augment_prob = prob
        print('Heavy Augmentation Probability set to:', self.heavy_augment_prob)

    def __getitem__(self, index):
        # Get the image and the circle parameters
        image_path, px, py, pr, ix, iy, ir = self.data[index]
        
        image = load_image(image_path)
        
        w, h = image.size
                        
        w_ratio = 320/w
        h_ratio = 240/h

        # Resize image
        if self.train_on_image:
            random_degree = np.random.choice([1,2,3,4,5,6])
            if random_degree == 1:
                image = image.resize((320, 240), Image.Resampling.NEAREST)
            elif random_degree == 2:
                image = image.resize((320, 240), Image.Resampling.BILINEAR)
            elif random_degree == 3:
                image = image.resize((320, 240), Image.Resampling.BICUBIC)
            elif random_degree == 4:
                image = image.resize((320, 240), Image.Resampling.LANCZOS)
            elif random_degree == 5:
                image = image.resize((320, 240), Image.Resampling.HAMMING)
            else:
                image = image.resize((320, 240), Image.Resampling.BOX)
        else:
            image = image.resize((320, 240), Image.Resampling.NEAREST)
        
        px *= w_ratio
        py *= h_ratio
        pr *= max(w_ratio, h_ratio)
        
        ix *= w_ratio
        iy *= h_ratio
        ir *= max(w_ratio, h_ratio)


        # Data augmentation
        if self.augment:
            
            if self.train_on_image:
                cval = random.randint(0, 255)
                mode = random.choice(["constant", "edge"])
                order = np.random.choice([0,1,3])
            else:
                cval = 0
                mode = "constant"
                order = 0
                           
            
            # random horizontal flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT) 
                px = 320 - px
                ix = 320 - ix
            
            # random scaling
            if random.random() < 0.5:
                random_scale = random.uniform(0.7, 1.4)
                aug = iaa.Affine(scale=random_scale, mode=mode, cval=cval, order=order)
                img_np = aug(images = np.expand_dims(np.expand_dims(np.uint8(image), axis=0), axis=-1))
                image = Image.fromarray(np.uint8(img_np[0, :, :, 0]))
                px = 160 - random_scale * (160 - px)
                py = 120 - random_scale * (120 - py)
                pr = random_scale * pr
                ix = 160 - random_scale * (160 - ix)
                iy = 120 - random_scale * (120 - iy)
                ir = random_scale * ir
            
            # random translation
            if random.random() < 0.5:
                random_tx = random.randint(-50, 50)
                random_ty = random.randint(-50, 50)
                aug = iaa.Affine(translate_px={"x" : random_tx, "y" : random_ty}, mode=mode, cval=cval, order=order)
                img_np = aug(images = np.expand_dims(np.expand_dims(np.uint8(image), axis=0), axis=-1))
                image = Image.fromarray(np.uint8(img_np[0, :, :, 0]))
                px = px + random_tx
                py = py + random_ty
                ix = ix + random_tx
                iy = iy + random_ty
            
            # random rotation
            if random.random() < 0.5:
                random_rot = random.randint(-45, 45)
                aug = iaa.Affine(rotate=random_rot, mode=mode, cval=cval, order=order)
                img_np = aug(images = np.expand_dims(np.expand_dims(np.uint8(image), axis=0), axis=-1))
                image = Image.fromarray(np.uint8(img_np[0, :, :, 0]))
                rot_radian = (random_rot * math.pi)/180
                px_new = (px - 160) * math.cos(-rot_radian) + (py - 120) * math.sin(-rot_radian) + 160
                py_new = (py - 120) * math.cos(-rot_radian) - (px - 160) * math.sin(-rot_radian) + 120
                ix_new = (ix - 160) * math.cos(-rot_radian) + (iy - 120) * math.sin(-rot_radian) + 160
                iy_new = (iy - 120) * math.cos(-rot_radian) - (ix - 160) * math.sin(-rot_radian) + 120
                px = int(px_new)
                py = int(py_new)
                ix = int(ix_new)
                iy = int(iy_new)
            
            if self.train_on_image and random.random() < self.heavy_augment_prob:
                if random.random() < 0.5:
                    random_choice = np.random.choice([1,2,3,4,5])
                    if random_choice == 1:   
                        # sharpening
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.DETAIL)
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.SHARPEN)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.EDGE_ENHANCE)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                        else:
                            aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 2:
                        #blurring
                        random_degree = np.random.choice([1,2,3,4,5])
                        if random_degree == 1:
                            image = image.filter(ImageFilter.GaussianBlur())
                        elif random_degree == 2:
                            image = image.filter(ImageFilter.BLUR)
                        elif random_degree == 3:
                            image = image.filter(ImageFilter.SMOOTH)
                        elif random_degree == 4:
                            image = image.filter(ImageFilter.SMOOTH_MORE)
                        else:
                            aug = iaa.MotionBlur(k=(3, 5))
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 3:
                        #noise
                        if random.random() < 0.5:
                            aug = iaa.AdditiveGaussianNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                        else:
                            aug = iaa.AdditiveLaplaceNoise(scale=5)
                            img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                            image = Image.fromarray(img_np[0])
                    elif random_choice == 4:
                        # downscaling and upscaling
                        divider = random.random() + 1.1
                        cw, ch = image.size
                        new_cw = int(cw/divider)
                        new_ch = int(ch/divider)
                        
                        first_choice = np.random.choice([1,2,3,4,5,6])
                        if first_choice == 1:
                            image = image.resize((new_cw, new_ch), Image.NEAREST)
                        elif first_choice == 2:
                            image = image.resize((new_cw, new_ch), Image.BILINEAR)
                        elif first_choice == 3:
                            image = image.resize((new_cw, new_ch), Image.BICUBIC)
                        elif first_choice == 4:
                            image = image.resize((new_cw, new_ch), Image.LANCZOS)
                        elif first_choice == 5:
                            image = image.resize((new_cw, new_ch), Image.HAMMING)
                        else:
                            image = image.resize((new_cw, new_ch), Image.BOX)
                        
                        second_choice = np.random.choice([1,2,3,4,5,6])
                        if second_choice == 1:
                            image = image.resize((cw, ch), Image.NEAREST)
                        elif second_choice == 2:
                            image = image.resize((cw, ch), Image.BILINEAR)
                        elif second_choice == 3:
                            image = image.resize((cw, ch), Image.BICUBIC)
                        elif second_choice == 4:
                            image = image.resize((cw, ch), Image.LANCZOS)
                        elif second_choice == 5:
                            image = image.resize((cw, ch), Image.HAMMING)
                        else:
                            image = image.resize((cw, ch), Image.BOX)
                    else:
                        #JPEG compression
                        aug = iaa.JpegCompression(compression=(50, 80))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])                        
                             
                if random.random() < 0.5: # random contrast change
                    random_degree = np.random.choice([1,2,3,4,5,6])                
                    if random_degree == 1:
                        aug = iaa.GammaContrast((0.5, 2.0))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 2:
                        aug = iaa.LinearContrast((0.4, 1.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 3:
                        aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    elif random_degree == 4:
                        aug = iaa.LogContrast(gain=(0.6, 1.4))
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])            
                    elif random_degree == 5:    
                        aug = iaa.pillike.EnhanceBrightness()
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
                    else:
                        aug = iaa.Alpha((0.1, 1.0), iaa.HistogramEqualization())
                        img_np = aug(images = np.expand_dims(np.array(image), axis=0))
                        image = Image.fromarray(img_np[0])
            
                    
        if self.input_transform is not None:
            image = self.input_transform(image)
        
        data = {'image': image, 'circle_param': torch.tensor([px/320, py/240, pr/160, ix/320, iy/240, ir/200])} # x / w, y / h, pr / (0.8 * sqrt(w^2 + h^2)), ir / sqrt(w^2 + h^2)

        return data

    def __len__(self):
        return self.length