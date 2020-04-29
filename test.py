from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

classes = ['barbarian', 'bard', 'cleric', 'druid',
'fighter', 'mage', 'monk', 'paladin', 'rogue',
'sorcerer', 'warlock', 'wizard']
data_type = 'test'
image_type = 'jpg'
number_of_images = 8
sets = 2
all_train_pics = 0
all_valid_pics = 0
all_test_pics = 0

PATH = 'dnd_classes/'
samples_dir = os.path.join(PATH, 'samples')
train_dir = os.path.join(PATH, 'train')
valid_dir = os.path.join(PATH, 'valid')
test_dir = os.path.join(PATH, 'test')
batch_size= 128
print('Counting..')
for i in range(0, len(classes)):
    
    curr_class = classes[i]
    class_train_dir = os.path.join(train_dir, curr_class)
    class_valid_dir = os.path.join(valid_dir, curr_class)
    class_test_dir =  os.path.join(test_dir, curr_class)
    
    total_pics = len(os.listdir(class_train_dir))
    all_train_pics +=total_pics
    print('Total ' + curr_class + ' training images:', total_pics)
    total_pics = len(os.listdir(class_valid_dir))
    all_valid_pics +=total_pics
    print('Total ' + curr_class + ' validation images:', total_pics)
    total_pics = len(os.listdir(class_test_dir))
    all_test_pics +=total_pics
    print('Total ' + curr_class + ' test images:', total_pics)
    print('-----')

print('Total training pics:', all_train_pics)
print('Steps per epoch', int(all_train_pics/batch_size))
print('-----')
print('Total valid pics:', all_valid_pics)
print('Steps per epoch', int(all_valid_pics/batch_size))
print('-----')
print('Total test pics:', all_test_pics)
print('Steps per epoch', int(all_test_pics/batch_size))
print('-----')

def data_amount(path_to_dir):
    total_data_amount = 0
    for i in range(0, len(classes)):
        class_dir = os.path.join(path_to_dir, classes[i])
        total_data_amount += len(os.listdir(class_dir))
    return total_data_amount

batch_size= 128
epochs = 5
IMG_H = 224
IMG_W = 224
steps_per_epoch = int(data_amount(train_dir)/batch_size)

img_gen = ImageDataGenerator(
                        rescale=1./255, 
                        horizontal_flip=True,
                        rotation_range = 90,
                        width_shift_range = 0.15,    
                        height_shift_range = 0.15,
                        shear_range = 0.15, 
                        zoom_range = 0.5,
                        )

def data_gen(dir):
    return img_gen.flow_from_directory(
                        batch_size = batch_size,
                        target_size=(IMG_H, IMG_W),
                        directory = dir,
                        shuffle = True,
                        class_mode = 'binary',
                        )
          
train_data = data_gen(train_dir)
valid_data = data_gen(samples_dir)
test_data = data_gen(test_dir)

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
model.fit_generator(train_data, steps_per_epoch=10, validation_data=valid_data, validation_steps=5, epochs=epochs, verbose=2)

#Finally, save the model
model.save('testual_mn_dnd_model.h5')