'''This script goes along the blog post
"Building powerful image classification models using very little data"
```
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

# from keras.models import load_weights


import numpy as np
from os import listdir, walk

import matplotlib
# Prevents "TclError: no display name and...":
matplotlib.use('Agg') 
# But "matplotlib.use() must be called *before* matplotlib.pyplot"
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 512, 512


is_Opti = True # Is optimized(4 classes) dataset being used?
if is_Opti:
    DatasetDir = './Datasets/Opti_SCV/'
    # ResultsDir = './Results/' + 'Opti_SCV/'

    ResultsDir = './Results/' + 'O_SCV_RedAdamDO/'

    # ResultsDir = './Results/' + 'O_SCV_E25_DAExpVal/'

    # DatasetDir = './Datasets/O_Multi/'
    # ResultsDir = './Results/' + 'O_Multi/'


    nb_classes = 4
else:
    # DatasetDir = './Datasets/SCV/'
    # ResultsDir = './Results/' + 'SCV/'

    # DatasetDir = './Datasets/SCV_Multi/'
    # ResultsDir = './Results/' + 'SCV_Multi_ExpDA/'

    DatasetDir = './Datasets/SCV/'
    ResultsDir = './Results/' + 'SCV_RedAdamDO/'

    nb_classes = 6

epochs = 15
batch_size = 32

FoldDirs = listdir(DatasetDir)
FoldDirs = FoldDirs[3:4] # select fewer folds

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
        x = Dropout(0.1)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = keras.Model(inputs=densenet.input, outputs=predictions)
    
    # ReducedAdam = keras.optimizers.Adam(lr=0.0001)
    # model.compile(loss='categorical_crossentropy',
    #             optimizer=ReducedAdam,
    #             metrics=['accuracy'])

    # model.load_weights('./Results/SCV_RedAdamDO/fold4_w.h5')
    model.load_weights('./Results/O_SCV_RedAdamDO/fold3_O_w.h5')

    ##################################################
    # Plain test_datagen: 
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # Create confusion matrix based on validation_generator
    CM_Name = 'Corrected_CM_' + Fold + '.txt'
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