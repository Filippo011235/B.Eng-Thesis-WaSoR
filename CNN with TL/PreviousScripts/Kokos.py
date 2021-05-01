from os import listdir
import numpy as np

from PIL import Image

from sklearn.utils import shuffle
from keras.utils import np_utils


def ExtractLabel(ImgName):
    """ Extract plastic class label from Image Name and return it"""
    # Each img has name notation "*****a0X*" where X is PlasticType
    PlasticType = ImgName[7] 
    return {
        '1': 0, # PET
        '2': 1, # HDPE
        '4': 2, # LDPE
        '5': 3, # PP
        '6': 4, # PS
        '7': 5, # Other
    }[PlasticType]


def PrepareDataset(TestOrTrainingPath, ImgSize, No_Classes):
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