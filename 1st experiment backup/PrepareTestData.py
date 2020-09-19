from os import listdir, rename

NoOfMaterials = 4
DirNames = ("a01/", "a02/", "a05/", "a06/")
TestDirNames = ("a01 test/", "a02 test/", "a05 test/", "a06 test/")
DirPostProcessing = "./PostProcessing Data/"
TestDatasetRatio = 0.1

if __name__ == "__main__":

    input("Press a key to begin: \n")
    
    for Material in range(NoOfMaterials):
        print("Material no.: ", Material)
        CurrentMatDir = DirPostProcessing + DirNames[Material]
        DestMatDir = DirPostProcessing + TestDirNames[Material]
        ImgListing = listdir(CurrentMatDir)

        NoOfImages = len(ImgListing)
        NoOfTestImg = TestDatasetRatio * NoOfImages

        for ImgNumber in range(int(NoOfTestImg)):
            CurrentFilePath = CurrentMatDir + ImgListing[ImgNumber]
            NewFilePath = DestMatDir + ImgListing[ImgNumber]
            rename(CurrentFilePath, NewFilePath)
        
