from os import listdir, scandir

# Get dir names and print them, just to be sure ;) 
SubfoldersList = [f.path for f in scandir('./') if f.is_dir()]
SubfoldersList = SubfoldersList[1:7] # ./a01 <-> a07
print(SubfoldersList)

# # Move data into ClassDivisionDirs
# for IdxDir in IdxDivisionDirs:
#     ImgList = listdir(IdxDir)
#     print(ImgList)
#     for Img in ImgList:
#         # if Img not "dekstop.ini":
#         ClassDirFromName = Img[5:8]
#         ImgIdxDir = IdxDir + "/" + Img
#         ImgClassDir = "./" + ClassDirFromName + "/" + Img
#         rename(ImgIdxDir, ImgClassDir)

# # ImgDir = './Raw from Wadaba/'
# ImgDir = './1 - 100/'
# ImgList = listdir(ImgDir)

# for ImgName in ImgList:
#     # if "c2" in ImgName:
#     #     remove(ImgDir + ImgName)
#     if "desktop" in ImgName:
#         print(ImgName)