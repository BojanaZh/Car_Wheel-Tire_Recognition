# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:38:56 2018

@author: Bojana
"""
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import glob

# load json and create model
json_file = open('/model/model_learn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model/model_learn.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

image_list = []
for filename in glob.glob('dataset/noTr/*.*'):
    im = image.load_img(filename, target_size = (32, 32))
    image_list.append(im)

print("Images Loaded")

count = 0
for test_image in image_list:
    
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict_proba(test_image)
    print(result)

    #training_set.class_indices
    if result[0][0] == 1:
        prediction = 'NoTire'
    else:
            prediction = 'Tire'
            count += 1
    


    
print(count)
