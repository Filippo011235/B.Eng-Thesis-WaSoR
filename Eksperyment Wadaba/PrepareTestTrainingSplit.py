from os import listdir, rename

NoOfMaterials = 4
DirPostProcessing = "./PostProcessing Data/"
DirNames = ("a01/", "a02/", "a05/", "a06/")
TestDirName = DirPostProcessing + "Test/"
TrainingDirName = DirPostProcessing + "Training/"
TestDatasetRatio = 0.1

if __name__ == "__main__":

    input("Press a key to begin: \n")
    
    for Material in range(NoOfMaterials):
        print("Material no. ", Material)
        CurrentMatDir = DirPostProcessing + DirNames[Material]
        #DestMatDir = DirPostProcessing + TestDirNames[Material]
        ImgListing = listdir(CurrentMatDir)

        NoOfImages = len(ImgListing)
        NoOfTestImg = TestDatasetRatio * NoOfImages


        # Doesn't work correctly :(
        # at each class there is a one item split between Train, Test
        # Index Train - Test
        # 06 320 - 320
        # 09 1680 - 720
        # 15 832 - 1248
        # 21 1100 - 1100
        # I just corrected it manually in a terminal

        # Designate TestDatasetRatio part of img for Test
        for ImgNumber in range(int(NoOfTestImg)):
            CurrentFilePath = CurrentMatDir + ImgListing[ImgNumber]
            NewFilePath = TestDirName + ImgListing[ImgNumber]
            rename(CurrentFilePath, NewFilePath)
        
        # Designate rest for Training
        for ImgNumber in range(int(NoOfTestImg), NoOfImages):
            CurrentFilePath = CurrentMatDir + ImgListing[ImgNumber]
            NewFilePath = TrainingDirName + ImgListing[ImgNumber]
            rename(CurrentFilePath, NewFilePath)

