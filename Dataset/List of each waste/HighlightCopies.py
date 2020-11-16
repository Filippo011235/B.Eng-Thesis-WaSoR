from os import listdir

ImgList = listdir("./")

CodeNamesArr = []
RepeatsArr = []

for ImgName in ImgList:
    ImgCodeName = ImgName[-10:]
    if ImgCodeName in ImgName:
        print(ImgName)

# for ImgName in ImgList:
#     ImgClass = 