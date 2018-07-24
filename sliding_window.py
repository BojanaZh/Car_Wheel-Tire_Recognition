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
from keras.models import load_model


# # Load weights into new model
model = load_model("checkpoints/model3-045.h5")
print("Loaded model from disk")


for img_no in range(1, 6):
    img_no = str(img_no)
    test_image = image.load_img('Images_from_video/out' + img_no + '.png')
    #test_image =  image.load_img('\dataset\learning_set\out' + img_no + '.jpg')
    x, y = test_image.size

    test_image = test_image.resize((round(x/3), round(y/3)), resample = BICUBIC)
    cv_image = np.array(test_image, dtype="float")
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(test_image)
    x, y = test_image.size

    prozorec1 = 64
    prozorec2 = 120
    prozorec3 = 150

    for i in range(0, x - prozorec3, 15):
        for j in range(0, y - prozorec3, 15):

            img1 = test_image.crop((i, j, i + prozorec1, j + prozorec1))
            img1 = img1.resize((32, 32), resample=BICUBIC)
            img1 = np.expand_dims(img1, axis = 0)
            img1 = np.array(img1, dtype="float") / 255.0 # napraj go ednas ova pogore a ne sto pati dole

            result1 = model.predict(img1)
            img2 = test_image.crop((i, j, i + prozorec2, j + prozorec2))
            img2 = img2.resize((32, 32), resample=BICUBIC)
            img2 = np.expand_dims(img2, axis = 0)
            img2 = np.array(img2, dtype="float") / 255.0 # napraj go ednas ova pogore a ne sto pati dole
            result2 = model.predict(img2)

            img3 = test_image.crop((i, j, i + prozorec3, j + prozorec3))
            img3 = img3.resize((32, 32), resample=BICUBIC)
            img3 = np.expand_dims(img3, axis = 0)
            img3 = np.array(img3, dtype="float") / 255.0 # napraj go ednas ova pogore a ne sto pati dole

            result3 = model.predict(img3)

            argmax = np.argmax(result1[0])
            if argmax==0 and result3[0][argmax] > 0.95:
                rect = patches.Rectangle((i,j), prozorec1, prozorec1,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)

            argmax = np.argmax(result2[0])
            if argmax==0 and result2[0][argmax] > 0.95:
                rect = patches.Rectangle((i,j), prozorec2, prozorec2,linewidth=1,edgecolor='g',facecolor='none')
                ax.add_patch(rect)


            argmax = np.argmax(result3[0])
            if argmax==0 and result3[0][argmax]>0.95:
                rect = patches.Rectangle((i,j), prozorec3, prozorec3,linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rect)



    plt.axis("off")
    fig.savefig('result_images/img' + img_no + '.png')

print("FINISHED")
