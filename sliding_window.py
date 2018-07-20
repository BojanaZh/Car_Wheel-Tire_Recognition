#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:42:28 2018

@author: bojana
"""

# Importing libraries and packages
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL.Image import BICUBIC

# Load json and create model
json_file = open('model/model_learn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model/model_learn.h5")
print("Loaded model from disk")

# Evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(test_set, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


for img_no in range(1, 2):
    img_no = str(img_no)
    test_image = image.load_img('Images_from_video/out' + img_no + '.png')
    #test_image = image.load_img('\dataset\learning_set\out' + img_no + '.jpg')
    x, y = test_image.size

    test_image = test_image.resize((round(x/3), round(y/3)), resample = BICUBIC)

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(test_image)
    x, y = test_image.size

    prozorec1 = 64
    prozorec2 = 120
    prozorec3 = 150
    

    #for i in range(0, x - prozorec3, 15):
        #for j in range(y//3, y - prozorec3, 15):
    for i in range(0, x - prozorec3, 15):
        for j in range(0, y - prozorec3, 15):   
        
            img1 = test_image.crop((i, j, i + prozorec1, j + prozorec1))
            img1 = img1.resize((32, 32), resample=BICUBIC)
            img1 = np.expand_dims(img1, axis = 0)
            result1 = loaded_model.predict(img1)

            img2 = test_image.crop((i, j, i + prozorec2, j + prozorec2))
            img2 = img2.resize((32, 32), resample=BICUBIC)
            img2 = np.expand_dims(img2, axis = 0)
            result2 = loaded_model.predict(img2)

            img3 = test_image.crop((i, j, i + prozorec3, j + prozorec3))
            img3 = img3.resize((32, 32), resample=BICUBIC)
            img3 = np.expand_dims(img3, axis = 0)
            result3 = loaded_model.predict(img3)

            #training_set.class_indices
            if result1[0][0] == 0:
                # Create a Rectangle patch
                rect = patches.Rectangle((i,j), prozorec1, prozorec1,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)

            if result2[0][0] == 0:
                # Create a Rectangle patch
                rect = patches.Rectangle((i,j), prozorec2, prozorec2,linewidth=1,edgecolor='g',facecolor='none')
                ax.add_patch(rect)

            if result3[0][0] == 0:
                # Create a Rectangle patch
                rect = patches.Rectangle((i,j), prozorec3, prozorec3,linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rect)
            


    plt.axis("off")
    
    fig.savefig('result_images/img' + img_no + '.png')
    
print("FINISHED")