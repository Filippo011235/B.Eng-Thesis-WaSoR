'''This script goes along the blog post
"Building powerful image classification models using very little data"
``` example required data structure:
data/
    train/
        dogs/
            dog001.jpg
            ...
        cats/
            cat001.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            ...
        cats/
            cat001.jpg
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
from sklearn.metrics import confusion_matrix

import numpy as np
from os import listdir, walk

import matplotlib
# Prevents "TclError: no display name and...":
matplotlib.use('Agg') 
# But "matplotlib.use() must be called *before* matplotlib.pyplot"
import matplotlib.pyplot as plt

# Dimensions of our images.
img_width, img_height = 512, 512


is_Opti = True # Is optimized(4 classes) dataset being used?
if is_Opti:
    # Various experiments:

    # DatasetDir = './Datasets/Opti_SCV/'
    # ResultsDir = './Results/' + 'Opti_SCV/'

    # ResultsDir = './Results/' + 'O_SCV_RedAdamDO/'

    # ResultsDir = './Results/' + 'O_SCV_E25_DAExpVal/'

    DatasetDir = './Datasets/O_Multi/'
    ResultsDir = './Results/' + 'O_Multi/'


    nb_classes = 4
else:
    # Various experiments:

    # DatasetDir = './Datasets/SCV/'
    # ResultsDir = './Results/' + 'SCV/'

    # DatasetDir = './Datasets/SCV_Multi/'
    # ResultsDir = './Results/' + 'SCV_Multi_ExpDA/'

    # DatasetDir = './Datasets/SCV/'
    # ResultsDir = './Results/' + 'SCV_RedAdamDO/'

    nb_classes = 6

epochs = 15
batch_size = 32

FoldDirs = listdir(DatasetDir)
FoldDirs = FoldDirs[3:] # select fewer folds

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

#  1. Instantiate a base model and load pre-trained weights into it.
densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(img_width,img_height,3)
)
#  2. Freeze all layers in the base model by setting trainable = False.
densenet.trainable = False
for layer in densenet.layers:
    layer.trainable = False

# Train model for each fold dir available.
for Fold in FoldDirs:
    # Basic set-up
    train_data_dir = DatasetDir + Fold + '/train'
    validation_data_dir = DatasetDir + Fold + '/validation'
    # Number of samples from given directory
    nb_train_samples = sum([len(files) for r, d, files in walk(train_data_dir)])
    nb_validation_samples = sum([len(files) for r, d, files in walk(validation_data_dir)])

    ##################################################
    # 3. Create a new model on top of the output of one 
    # (or several) layers from the base model.    
    x = densenet.output
    x = Flatten()(x)
    layer_units = [64]
    for num_units in layer_units:
        x = Dense(num_units, activation='relu')(x)
        x = Dropout(0.2)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = keras.Model(inputs=densenet.input, outputs=predictions)
    
    ReducedAdam = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                optimizer=ReducedAdam,
                metrics=['accuracy'])
    
    model.summary()

    ##################################################
    # Dataset generators:
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=15)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    # Plain test_datagen: 
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # test_datagen same as train_datagen: 
    # test_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     rotation_range=45,
    #     horizontal_flip=True,
    #     vertical_flip=True)
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    ##################################################
    # Actual trainig
    ModelHistory = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    
    ##################################################
    # Gathering and visualizing results

    if is_Opti: # Optimized classes or not? From beginning of the script
        model.save_weights(ResultsDir + Fold + '_O_w.h5') # Save weights

        # Prepare for Accuracy and Loss plots
        AccChartPath = ResultsDir + Fold + '_O_Acc.png'
        LossChartPath = ResultsDir + Fold + '_O_Loss.png'
    else:
        model.save_weights(ResultsDir + Fold + '_w.h5') 

        AccChartPath = ResultsDir + Fold + '_Acc.png'
        LossChartPath = ResultsDir + Fold + '_Loss.png'

    # Visualizing metrics and final score
    TrainLoss = ModelHistory.history['loss']
    ValLoss = ModelHistory.history['val_loss']
    TrainAcc = ModelHistory.history['accuracy']
    ValAcc = ModelHistory.history['val_accuracy']
    XAxis = range(1, epochs+1)

    acc_per_fold.append(ValAcc[-1] * 100) # in %
    loss_per_fold.append(ValLoss[-1])
    # print('-------------------------------')
    # print(Fold)
    # print(acc_per_fold[-1])
    # print(loss_per_fold[-1])
    # print('-------------------------------')
    
    ####################
    # Plot Loss and Accuracy graphs
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

    ####################
    # Create confusion matrix based on validation_generator
    CM_Name = 'CM_' + Fold + '.txt'
    cm_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)
    probabilities = model.predict_generator(generator=cm_generator)
    y_true = cm_generator.classes
    y_pred = np.argmax(probabilities, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    with open(ResultsDir + CM_Name, 'w') as f:
        print(cm, file=f)

    #######################################
    # Validation accuracy summary across folds, up to now
    with open(ResultsDir + 'Summary.txt', 'w') as f:
        # Provide average scores
        print('--------------------------------------------------------------', file=f)
        print('Score per fold', file=f)
        for i in range(0, len(acc_per_fold)):
            print('----------------------------------------------------------', file=f)
            print('> Fold ', i+1, " - Loss: ", round(loss_per_fold[i], 1), 
                    " - Accuracy: ", round(acc_per_fold[i], 1), "%", file=f)
        print('--------------------------------------------------------------', file=f)
        print('Average scores for all folds:', file=f)
        print("> Accuracy: ", round(np.mean(acc_per_fold),1), 
                " (+- ", round(np.std(acc_per_fold),1), ") %", file=f)
        print("> Loss: ", round(np.mean(loss_per_fold),1), file=f)
        print('---------------------------------------------------------------', file=f)