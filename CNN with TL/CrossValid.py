import numpy as np
from os import listdir
from shutil import move
from sklearn.model_selection import KFold

nb_folds = 10
# ClassDirs = ['/HDPE/', '/LDPE/', '/Other/', '/PET/', '/PP/', '/PS/']
ClassDirs = ['/HDPE/', '/LDPE/', '/Misc/', '/PETb/']
# ClassDirs = ['XXXX']
# FoldPath = './Dataset_KF/fold'
FoldPath = './Dataset_O_KF/fold'
FoldDirs = [FoldPath + str(fold_nb) for fold_nb in range(nb_folds)]
TrainDir = '/train'
ValDir = '/validation'

kf = KFold(n_splits=nb_folds, shuffle=True, random_state=1)

for Material in ClassDirs:
    DirPath = FoldDirs[0] + TrainDir + Material # FoldDirs[0] is arbitrary
    ImgListing = listdir(DirPath)
    nb_Imgs = len(ImgListing)
    ListingArr = np.array(range(nb_Imgs))
    i = 0 # FoldDir iterator
    for train_index, test_index in kf.split(ListingArr):
        print("TEST:", test_index)
        for Idx in test_index:
            print(ImgListing[Idx])
            ImgSource = FoldDirs[i] + TrainDir + Material + ImgListing[Idx]
            ImgDest = FoldDirs[i] + ValDir + Material + ImgListing[Idx]
            # print(ImgSource + " move to " + ImgDest)
            move(ImgSource, ImgDest)
        i += 1

