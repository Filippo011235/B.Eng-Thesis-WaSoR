from os import listdir, scandir
from shutil import copyfile
from PIL import Image

DatasetDir = './Dataset_h0_Multi'
# PostProcSize = (512,512) # pixels
RepeatClassIterations = [1, # PET
                        0, # HDPE
                        1, # LDPE 
                        1, # PP
                        1, # PS
                        0] # Other 
                        

# Get dir names and print them, just to be sure ;) 
ClassDirs = [f.path for f in scandir(DatasetDir) if f.is_dir()]
print(ClassDirs)
input("Ready?")

i = 0
for Dir in ClassDirs:
    # if i > 1: # Little accident, had to start from the PP
    ImgList = listdir(Dir)
    for Img in ImgList:
        ImgPath = Dir + "/" + Img
        CurrentImg = Image.open(ImgPath)
        # CurrentImg = CurrentImg.resize(PostProcSize)
        # CurrentImg.save(ImgPath)
        # CurrentImg.close()

        for k in range(RepeatClassIterations[i]):
            ImgDestPath = Dir + "/" + Img[:-4] + "_" + str(k) + ".jpg" 
            copyfile(ImgPath, ImgDestPath)
        # print("Finished ", Img)
    i += 1