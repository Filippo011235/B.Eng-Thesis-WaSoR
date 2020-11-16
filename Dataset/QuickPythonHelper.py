from os import listdir, rename, scandir


# Get dir names and print them, just to be sure ;) 
SubfoldersList = [f.path for f in scandir('./') if f.is_dir()]
IdxDivisionDirs = SubfoldersList[1:8] # ./1 <-> 340, without ./vscode
ClassDivisionDirs = SubfoldersList[8:-3] # ./a01 <-> a07
print(IdxDivisionDirs, "\n")
print(ClassDivisionDirs, "\n")

# Move data into ClassDivisionDirs
for IdxDir in IdxDivisionDirs:
    ImgList = listdir(IdxDir)
    for Img in ImgList:
        ClassDirFromName = Img[5:8]
        ImgIdxDir = IdxDir + "/" + Img
        ImgClassDir = "./" + ClassDirFromName + "/" + Img
        print(ImgIdxDir, " to ", ImgClassDir)
        # rename(ImgIdxDir, ImgClassDir)

# # ImgDir = './Raw from Wadaba/'
# ImgDir = './1 - 100/'
# ImgList = listdir(ImgDir)

# for ImgName in ImgList:
#     # if "c2" in ImgName:
#     #     remove(ImgDir + ImgName)
#     if "desktop" in ImgName:
#         print(ImgName)