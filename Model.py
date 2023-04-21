import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
import os

train_data='./data/train'
validate_data='./data/test'

train_data_gen = ImageDataGenerator (
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip= True,
    fill_mode='nearest'
)

validate_data_gen= ImageDataGenerator(rescale=1./255)

train_generate= train_data_gen.flow_from_directory(
    train_data,
    color_mode="grayscale",
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validate_generate= validate_data_gen.flow_from_directory(
    validate_data,
    color_mode="grayscale",
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

labels = ['angry', 'disgust', 'fear', 'happy','neutral', 'sad', 'surprise']

image, label= train_generate.__next__()

cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape = (48, 48, 1)))

cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))
cnn_model.add(Dropout(0.2))


cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(5, activation='softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
print(cnn_model.summary)

train_path = './data/train'
test_path = './data/test'


num_train_imgs = 0
for root, dir, files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dir, files in os.walk(test_path):
    num_test_imgs += len(files)

epochs=30
cnn_history = cnn_model.fit(train_generate, 
                            steps_per_epoch=num_train_imgs//32 , 
                            epochs=epochs, 
                            validation_data=validate_generate, 
                            validation_steps=num_test_imgs//32,
                            verbose=1, 
                            shuffle=True)

cnn_model.save('cnn_model.h5')