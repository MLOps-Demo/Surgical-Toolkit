# define and train model
import keras
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.engine.input_layer import Input
import yaml
import matplotlib.pyplot as plt
import json
from keras import optimizers

params = yaml.safe_load(open("params.yaml"))["training"]
img_shape = (224, 224, 3)
num_classes = params["num_classes"]
nb_epoch = params["nb_epoch"]
base_lr = params["base_lr"]

import numpy as np

train_data = np.load('data/train_data.npy')
train_label = np.load('data/train_label.npy')
val_data = np.load('data/val_data.npy')
val_label = np.load('data/val_label.npy')
test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')


def model_def():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Create your own input format
    keras_input = Input(shape=img_shape, name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(keras_input)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dense(64, activation='relu', name='fc4')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    model = Model(inputs=keras_input, outputs=x)
    return model

def schedule(epoch, decay=0.9):
    return base_lr * decay ** (epoch)

def train():
    model = model_def()
    print(model.summary())
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #.{epoch:02d}-{val_loss:.2f}
    callbacks = [keras.callbacks.ModelCheckpoint('saved-models/weights.h5',
                                                 verbose=1, save_best_only=True,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]

    # train model
    result = model.fit(train_data, train_label, epochs=nb_epoch, validation_data=(val_data, val_label),
                       callbacks=callbacks, verbose=1)

 
    
    test_score = model.evaluate(test_data, test_label)
    print('loss',test_score[0])
    print('accuracy',test_score[1])
    with open("scores.json", "w") as fd:
        json.dump({"loss": test_score[0], "accuracy": test_score[1]}, fd, indent=4)
    plt.figure(figsize=[8,6])
    plt.plot(result.history['loss'],'r',linewidth=3.0)
    plt.plot(result.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=12)
    plt.xlabel('Epochs ',fontsize=12)
    plt.ylabel('Loss',fontsize=12)
    plt.title('Loss Curves',fontsize=12)
    plt.savefig('loss.png')
    
    plt.figure(figsize=[8,6])
    plt.plot(result.history['accuracy'],'r',linewidth=3.0)
    plt.plot(result.history['val_accuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=12)
    plt.xlabel('Epochs ',fontsize=12)
    plt.ylabel('Accuracy',fontsize=12)
    plt.title('Accuracy Curves',fontsize=12)
    plt.savefig('accuracy.png')


if __name__ == '__main__':
    train()
