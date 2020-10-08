from os import listdir
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json

TrainingDataPath = "./PostProcessing Data/Training"
TestDataPath = "./PostProcessing Data/Test"
MyOwnTestDataPath =  "./PostProcessing Data/My own test data"

# CNN preparation

# done 
NbrOfEpochs = 3
ImgColRow = 120
ImgChannels = 3
NbrOfClasses = 4
NbrOfConvFilters = 64
KernelSize_5 = 5
KernelSize_9 = 9

# what about learning coefficient?
# Learning was carried out for a variable value of learning coefficient, 
# starting from 0.001 and decreasing every subsequent 4 epoch. 
# 1064 iterations for one epoch were considered.

# not sure
PoolSize = 2
# BatchSize = 512
BatchSize = 1 # temp


def EnvChecklist():
    """ Basic checkup is everything alright with libraries etc. """  
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

def ExtractLabel(ImgName):
    """ Extract plastic class label from Image Name and return it"""
    PlasticType = ImgName[7] # Each img has name notation "*****a0X*******" where X is PlasticType
    return {
        '1': 0, # PET
        '2': 1, # HDPE
        '5': 2, # PP
        '6': 3, # PS
    }[PlasticType]

def PrepareDataset(TestOrTrainingPath, ImgSize):
    ImgList = listdir(TestOrTrainingPath)
    ImgNbr = len(ImgList)

    # ImgMatrix = np.array
    # for Img in ImgList:
    #     ImgArray = np.array(Image.open(TestOrTrainingPath + "/" + Img)).flatten()
    #     print(ImgArray.shape) 
    #     np.append(ImgMatrix, ImgArray)
    ImgMatrix = np.array([np.array(Image.open(TestOrTrainingPath + "/" + Img)).flatten()
                    for Img in ImgList], 'f')

    # Create a np array containing labels of the data
    ImgLabels = np.ones((ImgNbr,), dtype = int)
    for i in range(ImgNbr):
        ImgLabels[i] = ExtractLabel(ImgList[i])
    
    FinalData, FinalLabels = shuffle(ImgMatrix, ImgLabels, random_state = 2)
    
    FinalData = FinalData.reshape(FinalData.shape[0], ImgSize, ImgSize, 3)
    FinalData = FinalData.astype('float32')
    FinalData /= 255

    # Convert class vectors to binary class matrices
    FinalLabels = np_utils.to_categorical(np.array(FinalLabels), NbrOfClasses)

    # Checkup
    # print("##########################################")
    # kokos = ImgMatrix[1].reshape(120,120, 3) #uintf8
    # kokos = ImgMatrix[1].reshape(120,120, 3)
    # plt.imshow(kokos)
    # plt.show()
    # print(FinalData.shape)
    # print(FinalLabels.shape)

    return [FinalData, FinalLabels]
    
###########################################
if __name__ == "__main__":

    X_train, Y_train = PrepareDataset(TrainingDataPath, ImgColRow)
    X_val, Y_val = PrepareDataset(TestDataPath, ImgColRow)

    # Checkup
    # print("Y_val print: ")
    # print(Y_val)
    # kokos = X_val[0].reshape(120,120, 3)
    # plt.imshow(kokos)
    # plt.show()

    model = Sequential()
    model.add(Convolution2D(NbrOfConvFilters, KernelSize_9,
                input_shape=(ImgColRow, ImgColRow, ImgChannels), padding='same' ))
    model.add(MaxPooling2D(PoolSize))
    model.add(Activation('relu'))
    model.add(Convolution2D(NbrOfConvFilters, KernelSize_5, padding = 'same')) 
    model.add(Activation('relu'))
    model.add(AveragePooling2D(PoolSize)) # padding valid/same
    model.add(Convolution2D(NbrOfConvFilters, KernelSize_5)) # padding valid
    model.add(Activation('relu'))
    model.add(AveragePooling2D(PoolSize)) # padding valid/same
    model.add(Flatten())
    model.add(Dense(64)) # Fully connected 64 x 10816
    model.add(Activation('relu'))
    model.add(Dense(NbrOfClasses)) # Fully connected 4 x 64
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])    
    model.summary()

    FitnessModel = model.fit(X_train, Y_train, batch_size= BatchSize, epochs= NbrOfEpochs, 
                        verbose= 1, validation_data= (X_val, Y_val))

    # Saving the model & weights
    # Serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("model.h5")
    print("Model has been saved to the disk")

    # Visualizing metrics and final score
    TrainLoss = FitnessModel.history['loss']
    ValLoss = FitnessModel.history['val_loss']
    TrainAcc = FitnessModel.history['accuracy']
    ValAcc = FitnessModel.history['val_accuracy']
    XAxis = range(NbrOfEpochs)
    
    plt.show(block = False)
    # fig, (PltLoss, PltAcc) = plt.subplot(1,2)
    # fig.suptitle('Loss and Accuaracy')
    # PltLoss.set_title("Loss")
    
    plt.figure(1)
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.plot(XAxis, TrainLoss, label = 'Train')
    plt.plot(XAxis, ValLoss, label = 'Val')
    plt.legend()

    plt.figure(2)
    plt.title("Acc")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(XAxis, TrainAcc, label = 'Train')
    plt.plot(XAxis, ValAcc, label = 'Val')
    plt.legend()
    
    print("Evaluation on validation dataset")
    score = model.evaluate(X_val, Y_val, verbose = 0)
    print('Test loss: ', score[0])
    print('Test acc: ', score[1])

    print("Prediction on new images")
    X_test, Y_test = PrepareDataset(MyOwnTestDataPath, ImgColRow)
    score = model.evaluate(X_test, Y_test, verbose = 0)
    print('Test loss: ', score[0])
    print('Test acc: ', score[1])
    print("Correct classes:")
    print(Y_test)
    print("Predict classes:")
    print(np.argmax(model.predict(X_test), axis = -1))
    # print(model.predict(X_test)) # more detailed
    # print(model.predict_classes(X_test)) # soon to be outdated
    
    plt.show() # Hold plots until closed
    

