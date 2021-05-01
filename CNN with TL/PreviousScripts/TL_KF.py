'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications import DenseNet121
from keras import layers

import numpy as np
from os import listdir, walk

import matplotlib
# Prevents "TclError: no display name and...":
matplotlib.use('Agg') 
# But "matplotlib.use() must be called *before* matplotlib.pyplot"
import matplotlib.pyplot as plt


# dimensions of our images.
img_width, img_height = 512, 512

is_Opti = False
if is_Opti:
    DatasetDir = './Dataset_O_KF/'
else:
    DatasetDir = './Dataset_KF/'

nb_classes = 6
epochs = 10
batch_size = 32

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)

#  1. Instantiate a base model and load pre-trained weights into it.
densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(img_width,img_height,3)
)
#  2. Freeze all layers in the base model by setting trainable = False.
densenet.trainable = False

FoldDirs = listdir(DatasetDir)
# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

for Fold in FoldDirs:
    # 3. Create a new model on top of the output of one 
    # (or several) layers from the base model.

    train_data_dir = DatasetDir + Fold + '/train'
    validation_data_dir = DatasetDir + Fold + '/validation'

    nb_train_samples = sum([len(files) for r, d, files in walk(train_data_dir)])
    nb_validation_samples = sum([len(files) for r, d, files in walk(validation_data_dir)])

    if is_Opti:
        AccChartPath = './Results/' + Fold + '_O_Acc.png'
        LossChartPath = './Results/' + Fold + '_O_Loss.png'
    else:
        AccChartPath = './Results/' + Fold + '_Acc.png'
        LossChartPath = './Results/' + Fold + '_Loss.png'

    
    inputs = keras.Input(shape=(img_width, img_height, 3))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = densenet(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    # A Dense classifier with a six units
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dense(32)(x)
    outputs = keras.layers.Dense(nb_classes)(x)
    model = keras.Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # model.summary()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    ModelHistory = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    if is_Opti:
        model.save_weights('./Weights/' + Fold + '_O_w.h5')
    else:
        model.save_weights('./Weights/' + Fold + '_w.h5')
    

    # Visualizing metrics and final score
    TrainLoss = ModelHistory.history['loss']
    ValLoss = ModelHistory.history['val_loss']
    TrainAcc = ModelHistory.history['accuracy']
    ValAcc = ModelHistory.history['val_accuracy']
    XAxis = range(1, epochs+1)

    acc_per_fold.append(ValAcc[-1] * 100)
    loss_per_fold.append(ValLoss[-1])
    print('-------------------------------')
    print(Fold)
    print(acc_per_fold[-1])
    print(loss_per_fold[-1])

    plt.figure(1)
    plt.clf()
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.plot(XAxis, TrainLoss, label='Train')
    plt.plot(XAxis, ValLoss, label='Val')
    plt.legend()
    plt.savefig(LossChartPath)

    plt.figure(2)
    plt.clf()
    plt.title("Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(XAxis, TrainAcc, label='Train')
    plt.plot(XAxis, ValAcc, label='Val')
    plt.legend()
    plt.savefig(AccChartPath)

# Provide average scores
print('--------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('----------------------------------------------------------')
    print('> Fold ', i+1, " - Loss: ", round(loss_per_fold[i], 1), 
            " - Accuracy: ", round(acc_per_fold[i], 1), "%")
print('--------------------------------------------------------------')
print('Average scores for all folds:')
print("> Accuracy: ", round(np.mean(acc_per_fold),1), 
        " (+- ", round(np.std(acc_per_fold),1), ") %")
print("> Loss: ", round(np.mean(loss_per_fold),1))
print('---------------------------------------------------------------')