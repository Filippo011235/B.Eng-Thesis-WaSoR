from os import listdir, rename
from os import scandir


SubfoldersList = [f.path for f in scandir('./') if f.is_dir()]
del SubfoldersList[-2:] # del iCloud Photos and Unknown
print(SubfoldersList) # just to be sure ;)




# OriginalNames = listdir('./Backup/')
# OriginalNames = listdir('./iCloud Photos/')

# for target_list in expression_list:
#     pass

# ObjectIdx = int(input("Idx of the 1st object: "))
# i = 0 

# while i < len(OriginalNames):
#     NoOfImages = input("No. of object images: ")
#     print("00 unknown | 01 PET | 02 PEHD | 03 PVC | 04 PELD | 05 PP | 06 PS | 07 Other")
#     TypeOfPlastic = input("Type of plastic(only 2nd number): 0")
#     j = 0
#     for j in range(int(NoOfImages)):
#         Designation = '0'+str(ObjectIdx) + '_a0'+TypeOfPlastic + 'h'+str(j) + ".JPG"  
#         print(OriginalNames[i], " to ", Designation)
#         rename('./iCloud Photos/'+OriginalNames[i], './iCloud Photos/'+Designation)
#         i += 1
#     ObjectIdx += 1 
    
