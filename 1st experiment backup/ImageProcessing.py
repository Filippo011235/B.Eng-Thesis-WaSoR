from os import listdir
from PIL import Image

NoOfMaterials = 4
DirNames = ("a01/", "a02/", "a05/", "a06/")
DirRaw = "./Raw from Wadaba/"
DirPostProcessing = "./PostProcessing Data/"
PostProcSize = (120,120) # pixels
#MaterialRatio = (16, 55, 52, 60)
MaterialAngle = (360/MaterialRatio[0], 360/MaterialRatio[1], 360/MaterialRatio[2], 360/MaterialRatio[3])

def ProcessImage(CurrentImage, j, RotAngle):
    return (CurrentImage.rotate(j * RotAngle)).resize(PostProcSize)
    
if __name__ == "__main__":

    for Material in range(NoOfMaterials):
        CurrentMatDir = DirRaw + DirNames[Material]
        RawImgListing = listdir(CurrentMatDir)

        boundary = 0

        for ImgName in RawImgListing:
            boundary = boundary + 1
            CurrentImg = Image.open(CurrentMatDir + ImgName)
            RotAngle = MaterialAngle[Material]
            for j in range(MaterialRatio[Material]):
                PostProcImg = ProcessImage(CurrentImg, j, RotAngle)
                PostProcImg.save(DirPostProcessing + DirNames[Material] + str(j) + ImgName)
            CurrentImg.close()
            if boundary >= 3:
                break 