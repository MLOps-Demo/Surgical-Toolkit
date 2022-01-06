import cv2
import os
import numpy as np
import keras
import tensorflow as tf
#import shutil
from keras.utils.np_utils import to_categorical

path_prefix = 'classification/'

class_list = os.listdir(path_prefix)
num_classes = len(class_list)

image_size = (224,224)

def pre_npy_data(setname):
    print('preparing',setname,'data....')
    data = []
    label = []
    for i in range(len(class_list)):
#        print(class_list[i])
        if class_list[i]=='Missing':
            class_label = 1
        elif class_list[i]=='No_missing':
            class_label = 0
        img_list = os.listdir(path_prefix+class_list[i]+'/'+setname+'/')
        for j in range(len(img_list)):
            img_name = path_prefix+class_list[i]+'/'+setname+'/'+img_list[j]
            img = cv2.imread(img_name)
#            print(img_name)
            img = cv2.resize(img,image_size)
            data.append(img)
            label.append(class_label)
    data = np.array(data,'float32')
    label = np.array(label)
    shuffle_id = np.arange(len(data))
    np.random.shuffle(shuffle_id)
    data = data[shuffle_id, :,:,:]
    label = label[shuffle_id]
    cat_label = keras.utils.to_categorical(label, num_classes)

    np.save('data/'+setname+'_data.npy', data)
    np.save('data/'+setname+'_label.npy', cat_label)


if __name__ == '__main__':
    pre_npy_data('train')
    pre_npy_data('val')
    pre_npy_data('test')


