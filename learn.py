#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:42:28 2018

@author: bojana
"""

# Importing libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from keras.layers.normalization import BatchNormalization
from noisy import noisy

#Constants
img_width, img_height = 32, 32
epochs = 50
batch_size = 16
input_shape = (img_width, img_height, 3)
page_dir_path  = 'dataset'
filepath="saved_weights/weights-improvement-v-{epoch:02d}-{val_acc:.2f}.hdf5"

#The model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', W_constraint=max_norm(3)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', W_constraint=max_norm(3)))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', W_constraint=max_norm(3)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', W_constraint=max_norm(3)))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32, activation='relu', W_constraint=max_norm(3)))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Load images
print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(page_dir_path)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    #image = cv2.resize(image, (32, 32))
    image = noisy("s&p", image)
    image = img_to_array(image)
    data.append(image)
    
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "Tr" else 0
    labels.append(label)
    

# Scaleing imges to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the dataset to train and test 
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, random_state=42)


# Convert the labels from integers to vectors
#trainY = to_categorical(trainY, num_classes=1)
#testY = to_categorical(testY, num_classes=1)


# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# Fit the data to the model
history_callback = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
	epochs=epochs, verbose=1, callbacks=callbacks_list)

# Extracting callbacks info
loss_history = history_callback.history["loss"]
val_history = history_callback.history["val_acc"]

# Saving loss history
numpy_loss_history = np.array(loss_history)
np.savetxt("history/clearSides_v8-{epoch:02d}-{val_acc:.2f}_loss_history.txt", numpy_loss_history, delimiter=",")
numpy_val_history = np.array(val_history)
np.savetxt("history/clearSides_v8-{epoch:02d}-{val_acc:.2f}_val_history.txt", numpy_val_history, delimiter=",")

# Serialize model to JSON
model_json = model.to_json()
with open("model/model_learn_v8.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model_learn_v8.h5")
print("Saved model to disk")

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), history_callback.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history_callback.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history_callback.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history_callback.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Tr")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot/plot")