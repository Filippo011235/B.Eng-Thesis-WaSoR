from os import listdir
from PIL import Image

NoOfMaterials = 4
DirNames = ("a01/", "a02/", "a05/", "a06/")
DirRaw = "./Raw from Wadaba/"
DirPostProcessing = "./PostProcessing Data/"
PostProcSize = (120,120) # pixels
MaterialRatio = (16, 55, 52, 60) # How many more images have to be created from original dataset
# Rotation angle needed to achieve MatRatio. Each pic will be rotated by this angle
MaterialAngle = (360/MaterialRatio[0], 360/MaterialRatio[1], 360/MaterialRatio[2], 360/MaterialRatio[3])

def ProcessImage(CurrentImage, j, RotAngle):
    return (CurrentImage.rotate(j * RotAngle)).resize(PostProcSize)
    
if __name__ == "__main__":

    input("Press Enter to begin")

    for Material in range(NoOfMaterials):
        print("Material no.: ", Material)
        CurrentMatDir = DirRaw + DirNames[Material]
        RawImgListing = listdir(CurrentMatDir)

        for ImgName in RawImgListing:
            print("Working on: ", ImgName)
            CurrentImg = Image.open(CurrentMatDir + ImgName)
            RotAngle = MaterialAngle[Material]
            for j in range(MaterialRatio[Material]):
                PostProcImg = ProcessImage(CurrentImg, j, RotAngle)
                # Create new name for image -> appendix rXX.jpg where XX is "j" from loop
                NewImgName = ImgName[:-4] + "r" + str(j) + ".jpg"
                PostProcImg.save(DirPostProcessing + DirNames[Material] + NewImgName)
            CurrentImg.close()

    input("Press Enter to continue...")