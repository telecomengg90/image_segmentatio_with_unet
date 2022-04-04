# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:38:57 2022

@author: Dell
"""

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


# arrange the train_images and masks together

backbone = "resnet34"
preprocess_input = sm.get_preprocessing(backbone) 

train_images = []
train_masks = []
image_directory = r"H:\python_practice\unet-master\data\membrane\train\aug_img"
mask_directory = r"H:\python_practicer\unet-master\data\membrane\train\aug_mask"

counter_image = 1000
for file in os.listdir(image_directory):
    if file.endswith(".png"):
        
        image_name = os.path.join(image_directory, file)
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        train_images.append(image)
        counter_image -=1
        print("Remaining_files: {}".format(counter_image))
    
    else:
        continue
counter_mask = 1000    
for file in os.listdir(mask_directory):
    if file.endswith(".png"):
        
        mask_name = os.path.join(mask_directory, file)
        mask = cv2.imread(mask_name, 0)
        train_masks.append(mask)
        counter_mask -=1
        print("Remaining_files: {}".format(counter_mask))
    
    else:
        continue
    
train_images = np.array(train_images)
train_masks = np.array(train_masks)         

x = train_images
y = train_masks

y = np.expand_dims(y, axis = 3)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.2, random_state = 42)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

model = sm.Unet(backbone, encoder_weights = "imagenet")

model.compile(optimizer = "adam", loss = sm.losses.bce_jaccard_loss, metrics = [sm.metrics.iou_score])

print(model.summary())



history = model.fit(x_train,
                    y_train,
                    batch_size = 8,
                    epochs = 10,
                    verbose = 1,
                    validation_data = (x_val,y_val)) # training here takews a lot of memory

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('membrane.h5')


from tensorflow import keras
model = keras.models.load_model('membrane.h5', compile=False)
#Test 
test_img = cv2.imread('membrane/test/0.png', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (256, 256))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)


prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='gray')
plt.imsave('membrane/test0_segmented.jpg', prediction_image, cmap='gray')








   
    