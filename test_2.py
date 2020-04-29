import os

classes = ['barbarian', 'bard', 'cleric', 'druid',
'fighter', 'mage', 'monk', 'paladin', 'rogue',
'sorcerer', 'warlock', 'wizard']
all_train_pics = 0
all_valid_pics = 0
all_test_pics = 0
data_string = 'DATA TREE FOR EVEN_BIGGER_MN_DND_MODEL.H5\n-----\n'
helper = 0

PATH = 'dnd_classes/'
samples_dir = os.path.join(PATH, 'samples')
train_dir = os.path.join(PATH, 'train')
valid_dir = os.path.join(PATH, 'valid')
test_dir = os.path.join(PATH, 'test')
batch_size= 128

def write_data_to_file(string_to_write):
    pass

for i in range(0, len(classes)):
    
    curr_class = classes[i]
    class_train_dir = os.path.join(train_dir, curr_class)
    class_valid_dir = os.path.join(valid_dir, curr_class)
    class_test_dir =  os.path.join(test_dir, curr_class)
    
    total_pics = len(os.listdir(class_train_dir))
    all_train_pics +=total_pics
    data_string += 'Total ' + curr_class + ' training images: ' + str(total_pics) + '\n'
    total_pics = len(os.listdir(class_valid_dir))
    all_valid_pics +=total_pics
    data_string += 'Total ' + curr_class + ' validation images: '+ str(total_pics) + '\n'
    total_pics = len(os.listdir(class_test_dir))
    all_test_pics +=total_pics
    data_string += 'Total ' + curr_class + ' test images: '+ str(total_pics) + '\n-----\n'

data_string += str('Total training pics: ' + str(all_train_pics)) + '\n'
helper = int(all_train_pics/batch_size)
data_string += str('Steps per epoch: ' +  str(helper) + '\n')
data_string += str('-----') + '\n'
helper = int(all_valid_pics/batch_size)
data_string += str('Total valid pics: ' + str(all_valid_pics)) + '\n'
data_string += str('Steps per epoch: ' +  str(helper)) + '\n'
data_string += str('-----') + '\n'
data_string += str('Total test pics: '+ str(all_test_pics)) + '\n'
helper = int(all_test_pics/batch_size)
data_string += str('Steps per epoch: ' + str(helper))  + '\n'
data_string += '-----' + '\n'

print(data_string)

docum = open('even_bigger_data_tree.txt', 'w')
docum.write(data_string)
docum.close()