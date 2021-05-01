import os
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.convolutional import \
        Convolution2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('Agg') # Prevents "TclError: no display name and..."

# Using Quick Test:
TrainingDataPath = "./PostProcessing Data/Quick test/Training"
TestDataPath = "./PostProcessing Data/Quick test/Test"
MyOwnTestDataPath =  "./PostProcessing Data/Quick test/My own test data"
DataPath = "./PostProcessing Data/Dataset"

# Using Full WaDaBa:
# TrainingDataPath = "./PostProcessing Data/Training"
# TestDataPath = "./PostProcessing Data/Test"
# MyOwnTestDataPath =  "./My own test data"

# CNN preparation:
# I'm sure of those:
No_Epochs = 10
ImgColRow = 120
ImgChannels = 3
No_Classes = 4
No_ConvFilters = 64
KernelSize_5 = 5
KernelSize_9 = 9

No_Folds = 10

# What about learning coefficient, optimizer?:
# Learning was carried out for a variable value of learning coefficient
# starting from 0.001 and decreasing every subsequent 4 epoch. 
# 1064 iterations for one epoch were considered.

# Still not sure about those:
PoolSize = 2
BatchSize = 64
# BatchSize = 1 # For quick tests

def ExtractLabel(ImgName):
    """ Extract plastic class label from Image Name and return it"""
    # Each img has name notation "*****a0X*" where X is PlasticType
    PlasticType = ImgName[7] 
    return {
        '1': 0, # PET
        '2': 1, # HDPE
        '5': 2, # PP
        '6': 3, # PS
        # '7': 4, # Other ``````````````````````````````
        # '4': 5, # LDPE`````````````````````````````
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
    ImgLabels = np.ones((ImgNbr,), dtype=int)
    for i in range(ImgNbr):
        ImgLabels[i] = ExtractLabel(ImgList[i])
    
    FinalData, FinalLabels = shuffle(ImgMatrix, ImgLabels, random_state=2)

    FinalData = FinalData.reshape(FinalData.shape[0], ImgSize, ImgSize, 3)
    FinalData = FinalData.astype('float32')
    FinalData /= 255

    # Convert class vectors to binary class matrices
    FinalLabels = np_utils.to_categorical(np.array(FinalLabels), No_Classes)

    return [FinalData, FinalLabels]
    

###########################################
if __name__ == "__main__":

    # X_train, Y_train = PrepareDataset(TrainingDataPath, ImgColRow)
    # X_val, Y_val = PrepareDataset(TestDataPath, ImgColRow)
    Input, Target = PrepareDataset(DataPath, ImgColRow)
    # Checkup
    # print("Y_val print: ")
    # print(Y_val)
    # kokos = X_val[0].reshape(120,120, 3)
    # plt.imshow(kokos)
    # plt.show()

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=No_Folds, shuffle=True)

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    fold_no = 1
    for train, test in kfold.split(Input, Target):

        model = Sequential()
        model.add(Convolution2D(No_ConvFilters, KernelSize_9,
                    input_shape=(ImgColRow, ImgColRow, ImgChannels), 
                    padding='same'))
        model.add(MaxPooling2D(PoolSize))
        model.add(Activation('relu'))
        model.add(Convolution2D(No_ConvFilters, KernelSize_5, 
                                    padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(PoolSize)) # padding: valid/same
        model.add(Convolution2D(No_ConvFilters, KernelSize_5)) # padding: valid
        model.add(Activation('relu'))
        model.add(AveragePooling2D(PoolSize)) # padding: valid/same
        model.add(Flatten())
        model.add(Dense(64)) # F.C. 64 x 10816
        model.add(Activation('relu'))
        model.add(Dense(No_Classes)) # F.C. 4 x 64
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy'])    
        model.summary()

        # Generate a print
        print('----------------------------------------------------------')
        print('Training for fold: ', fold_no)

        # FitnessModel = model.fit(Input[train], Target[train], 
        #                           batch_size= BatchSize, 
        #                           epochs= No_Epochs, verbose= 1, 
        #                           validation_data= (Input[test], Target[test]))
        FitnessModel = model.fit(Input[train], Target[train], 
                                    batch_size=BatchSize, 
                                    epochs=No_Epochs, 
                                    verbose=1,
                                    validation_data=(Input[test], Target[test]))
        
        # Generate generalization metrics


        scores = model.evaluate(Input[training], Target[training], verbose=0)


        scores = model.evaluate(Input[test], Target[test], verbose=0)
        print('Score for fold ', fold_no, " ", model.metrics_names[0], 
                " of ", round(scores[0], 1), "; ", model.metrics_names[1],
                " of ", round(scores[1]*100, 1), "%")
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Saving the model & weights
        # Serialize model to JSON
        # model_json = model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # Serialize weights to HDF5
        # model.save_weights("model.h5")
        # print("Model has been saved to the disk")

        # Visualizing metrics and final score
        TrainLoss = FitnessModel.history['loss']
        ValLoss = FitnessModel.history['val_loss']
        TrainAcc = FitnessModel.history['accuracy']
        ValAcc = FitnessModel.history['val_accuracy']
        XAxis = range(No_Epochs)
        
        # plt.show(block = False) # Useless on currently used PC
        
        plt.figure(1)
        plt.title("Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.plot(XAxis, TrainLoss, label='Train')
        plt.plot(XAxis, ValLoss, label='Val')
        plt.legend()
        plt.savefig("Loss" + str(fold_no) + ".png")

        plt.figure(2)
        plt.title("Acc")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.plot(XAxis, TrainAcc, label='Train')
        plt.plot(XAxis, ValAcc, label='Val')
        plt.legend()
        plt.savefig("Acc" + str(fold_no) + ".png")

        # print("Evaluation on validation dataset")
        # score = model.evaluate(X_val, Y_val, verbose = 0)
        # print('Test loss: ', score[0])
        # print('Test acc: ', score[1])

        # print("Prediction on new images")
        # X_test, Y_test = PrepareDataset(MyOwnTestDataPath, ImgColRow)
        # score = model.evaluate(X_test, Y_test, verbose = 0)
        # print('Test loss: ', score[0])
        # print('Test acc: ', score[1])
        # print("Correct classes:")
        # print(Y_test)
        # print("Predict classes:")
        # print(np.argmax(model.predict(X_test), axis = -1))
        # print(model.predict(X_test)) # more detailed
        # print(model.predict_classes(X_test)) # soon to be outdated
        
        # plt.show() # Hold plots until closed # useless on panamint
        
        # Increase fold number
        fold_no += 1

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