import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#Create paths to data
classes =   [
            'barbarian', 'bard', 'cleric', 'druid',
            'fighter', 'mage', 'monk', 'paladin', 'rogue',
            'sorcerer', 'warlock', 'wizard'
            ]
train_path = 'dnd_classes/train'
valid_path = 'dnd_classes/valid'
test_path = 'dnd_classes/test'

def data_amount(path):
    total_data_amount = 0
    for i in range(0, len(classes)):
        class_dir = os.path.join(path, classes[i])
        total_data_amount += len(os.listdir(class_dir))
    return total_data_amount

batch_size = 128
IMG_H = 224
IMG_W = 224
steps_per_epoch = int(data_amount(train_path)/batch_size)

#Prepare image data for the model
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(IMG_H, IMG_W), batch_size=batch_size)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(IMG_H, IMG_W), batch_size=batch_size)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(IMG_H, IMG_W), batch_size=batch_size)

#Import the ready MobileNet-model
mobile = tf.keras.applications.mobilenet.MobileNet()

#Create own model based on the MobileNet-model
x = mobile.layers[-6].output    #Take the MN layers, except the last 5
predictions = Dense(12, activation='softmax')(x)    #Create own output layer for 12 classes
model = Model(inputs=mobile.input, outputs=predictions)     #Now the model is Model not Sequential as in the prev exercises

# Let's train only the last five layers with our data (you can test different amounts of trainable layers)
# So we set all the other layers except the last 5 to non-trainable
for layer in model.layers[:-5]:
    layer.trainable=False

#Compile and fit the new model, steps_per_epoch = data_amount/batch_size
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches, validation_steps=10, epochs=5, verbose=2)

#Finally, save the model
model.save('even_bigger_mn_dnd_model.h5')