from os import listdir, remove

# ImgDir = './Raw from Wadaba/'
ImgDir = './1 - 100/'
ImgList = listdir(ImgDir)

for ImgName in ImgList:
    # if "c2" in ImgName:
    #     remove(ImgDir + ImgName)
    if "desktop" in ImgName:
        print(ImgName)
