from os import listdir, rename
from shutil import copy2
OriginalNames = listdir('./iCloud Photos/')

ObjectIdx = int(input("Idx of the 1st object: "))

i = 0 
while i < len(OriginalNames):
    NoOfImages = input("No. of object images: ")
    CodeName = input("Code-name: ")
    print("00 unknown | 01 PET | 02 PEHD | 03 PVC | 04 PELD | 05 PP | 06 PS | 07 Other")
    TypeOfPlastic = input("Type of plastic(only 2nd number): 0")
    j = 0
    # Copy first Img of Waste to "./List of each waste" dir, with CodeName
    RepresentiveImg = '0'+str(ObjectIdx) + '_a0'+TypeOfPlastic + 'h'+str(j) + "_"+CodeName + ".JPG"  
    copy2('./iCloud Photos/'+OriginalNames[i], './List of each waste/'+RepresentiveImg)
    # Rename all Images of Waste, in the current dir
    for j in range(int(NoOfImages)):
        Designation = '0'+str(ObjectIdx) + '_a0'+TypeOfPlastic + 'h'+str(j) + ".JPG"  
        print(OriginalNames[i], " to ", Designation)
        rename('./iCloud Photos/'+OriginalNames[i], './iCloud Photos/'+Designation)
        i += 1
    ObjectIdx += 1 
    
