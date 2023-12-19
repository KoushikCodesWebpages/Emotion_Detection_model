import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the emotion model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Learning rate schedule
initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

# Compile the model
emotion_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])

# Set the paths for training and validation data
train_dir = r"C:\Users\Koush\OneDrive\Desktop\virtusa_project\EX_TRAIN"
val_dir = r"C:\Users\Koush\OneDrive\Desktop\virtusa_project\EX_VAL"

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# For the training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=2,  # Adjusted batch size based on the small dataset
    color_mode="grayscale",
    class_mode='categorical'
)

# For the validation generator
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=2,
    color_mode="grayscale",
    class_mode='categorical'
)
# Train the model
emotion_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,  # Adjust the number of epochs as needed, consider reducing for a small dataset
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

from keras.models import load_model

# Assuming your Keras model is stored in a variable named 'model'
# and you've already trained it

# Save the model to an HDF5 file
emotion_model.save('emoji_model.h5')

