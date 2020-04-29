import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classes = ['barbarian', 'bard', 'cleric', 'druid',
'fighter', 'mage', 'monk', 'paladin', 'rogue',
'sorcerer', 'warlock', 'wizard']
data_type = 'train'
image_type = 'jpg'
number_of_images = 8
sets = 9

for z in range(0, len(classes)):
    
    character = classes[z]
    
    for y in range(0, sets):
        
        gen = ImageDataGenerator(
            rotation_range = 10 * y, 
            width_shift_range = 0.1,    
            height_shift_range = 0.1, 
            shear_range = 0.15 * y, 
            zoom_range = 0.1 * y,
            channel_shift_range = 10.0 * y, 
            horizontal_flip = True,
            brightness_range = [0.2,2.0],
            )
        
        for x in range(1,number_of_images + 1):
            image_path = 'dnd_classes/samples/' + character + '/' + character + '_' + str(x) + '.' + image_type
            image = np.expand_dims(plt.imread(image_path), 0)

            i = 0
            for batch in gen.flow(
                image, 
                batch_size=number_of_images,
                save_to_dir='dnd_classes/' + data_type + '/' + character,
                save_prefix=character,
                save_format='png'):
                
                i += 1
                if i > 5:
                    print('Set' + str(y +1) + '/' + str(sets) + ' ' + image_path + ' done')
                    print('------')
                    break

print('All done!')