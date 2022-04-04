# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:39:49 2022

@author: Dell
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import  random
import os
from scipy.ndimage import rotate

import albumentations as A

images_path = r"H:\python_practice\unet-master\unet-master\data\membrane\train\image"
mask_path = r"H:\python_practice\unet-master\unet-master\data\membrane\train\label"

img_augmented_path = r"H:\python_practice\unet-master\unet-master\data\membrane\train\aug_img"
mask_augmented_path = r"H:\python_practice\unet-master\unet-master\data\membrane\train\aug_mask"
images = []
masks = []

for file in os.listdir(images_path):
    images.append(os.path.join(images_path,file))
    
    
for mask in os.listdir(mask_path):
    masks.append(os.path.join(mask_path, file))
    
    
aug = A.Compose([A.VerticalFlip(p = 0.5),
                 A.RandomRotate90(p = 0.5),
                 A.HorizontalFlip(p = 1),
                 A.Transpose(p = 1),
                 A.GridDistortion(p = 1)])

images_to_generate = 1000

#now appy this augmentation to all the images in the directory 
# the aim is to increase the no of images
i = 1

while i< images_to_generate:
    
    no = random.randint(0, len(images)-1)
    image = images[no]
    mask = masks[no]
    original_image = io.imread(image)
    original_mask = io.imread(mask)
    
    augmented = aug(image = original_image,mask = original_mask)
    augmented_image = augmented["image"]
    augmented_mask = augmented["mask"]
    new_aug_image_path = img_augmented_path + r"\augmented_image" +"_"+str(i)+".png"
    new_aug_mask_path = mask_augmented_path + r"\augmented_mask" +"_"+str(i)+".png"
    
    io.imsave(new_aug_image_path ,augmented_image)
    io.imsave(new_aug_mask_path ,augmented_mask)
    
    i = i+1
    print("remaining_file: {}".format(1000-i))



    
    
    
    





    
    