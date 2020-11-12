from os import listdir, rename
from os import scandir
from shutil import copy2

DestDir = ("./List of each waste/")
# SubfoldersList = [f.path for f in scandir('./') if f.is_dir()]
SubfoldersList = ["./1 - 100"]

# del SubfoldersList[-3:] # del last dir: iCloud Photos, "List of...", and Unknown
# SubfoldersList = SubfoldersList[1:] # From "[N:]" directory onward

print(SubfoldersList) # just to be sure ;)

ArrOfWaste = [] # stores idx of each waste

for WasteFolder in SubfoldersList:
    ImgList = listdir(WasteFolder)
    for ImgName in ImgList:
        ImgIdx = ImgName[0 : 4] # Get Img first 4 letters(Idx e.g. "0321")
        if ImgIdx not in ArrOfWaste:
            ArrOfWaste.append(ImgIdx)
            ImgPath = WasteFolder + "/" + ImgName
            # add custom code-name to waste Img for easier look up
            InputText = "Name for " + ImgIdx + ": "
            WasteName = input(InputText)
            # Concatenate Directory, Img code(without ".JPG"), custom name and ".jpg" 
            DestDirAndName = DestDir + "/" + ImgName[:10] + "_" + WasteName + ".jpg"
            copy2(ImgPath, DestDirAndName)
