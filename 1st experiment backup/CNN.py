from os import listdir
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

TrainingDataPath = "./PostProcessing Data/Training"
TestDataPath = "./PostProcessing Data/Test" 

def Checklist():
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

def PrepareDataset(TestOrTrainingPath):
    ImgList = listdir(TestOrTrainingPath)
    ImgNbr = len(ImgList)

    # ImgMatrix = np.array
    # for Img in ImgList:
    #     ImgArray = np.array(Image.open(TestOrTrainingPath + "/" + Img)).flatten()
    #     print(ImgArray.shape) 
    #     np.append(ImgMatrix, ImgArray)
    ImgMatrix = np.array([np.array(Image.open(TestOrTrainingPath + "/" + Img)).flatten()
                    for Img in ImgList], 'f')
    ImgMatrix /= 255

    # Create a np array containing labels of the data
    ImgLabels = np.ones((ImgNbr,), dtype = int)
    for i in range(ImgNbr):
        ImgLabels[i] = ExtractLabel(ImgList[i])
    
    FinalData, FinalLabels = shuffle(ImgMatrix, ImgLabels, random_state = 2)
    ReturnData = [FinalData, FinalLabels]
    
    # Checkup
    # print("##########################################")
    kokos = ImgMatrix[1].reshape(120,120, 3)
    plt.imshow(kokos)
    plt.show()
    print(ReturnData[0].shape)
    print(ReturnData[1].shape)

#%%
# CNN preparation

# done 
NbrOfEpochs = 10 
ImgRow, ImgCol = 120, 120
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
BatchSize = 32 #?????
MaxPoolingSize = 2


#%%






###########################################
if __name__ == "__main__":
    PrepareDataset(TestDataPath)

    

