from os import listdir, scandir
from shutil import copyfile
from PIL import Image


# Get dir names and print them, just to be sure ;) 
FoldDirs = [f.path for f in scandir('./Datasets/O_Multi') if f.is_dir()]
print(FoldDirs)
print("--------------------------")

FoldDirs = FoldDirs[5:]
for Fold in FoldDirs:
    # # ['HDPE', 'LDPE', 'Other', 'PET', 'PP', 'PS'] 
    # TrainRepeatClass = [3, 3, 3, 0, 0, 2] \
    
    # ['HDPE', 'LDPE', 'Misc', 'PETb'] 
    TrainRepeatClass = [6, 7, 0, 1] 

    TrainDirs = [f.path for f in scandir(Fold + '/train') if f.is_dir()]
    print(TrainDirs)
    
    i = 0 # class iterator 
    for Dir in TrainDirs:
        # if i == 5: 
        # print("Multiply ", TrainDirs[i], " ", TrainRepeatClass[i], " times")
        ImgList = listdir(Dir)
        for Img in ImgList:
            ImgPath = Dir + "/" + Img
            for k in range(TrainRepeatClass[i]):
                ImgDestPath = Dir + "/" + Img[:-4] + "_" + str(k) + ".jpg" 
                copyfile(ImgPath, ImgDestPath)
                # print(ImgPath, " to ", ImgDestPath)
            # print("Finished ", Img)
        i += 1

    # # ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'Other'] 
    # ValRepeatClass = [3, 3, 0, 0, 2, 3] 

    # ['HDPE', 'LDPE', 'PETb', 'Misc'] 
    ValRepeatClass = [6, 7, 1, 0] 

    ValDirs = [f.path for f in scandir(Fold + '/validation') if f.is_dir()]
    print(ValDirs)

    i = 0 # class iterator 
    for Dir in ValDirs:
        # if i == 5: 
        # print("Multiply ", ValDirs[i], " ", ValRepeatClass[i], " times")
        ImgList = listdir(Dir)
        for Img in ImgList:
            ImgPath = Dir + "/" + Img
            for k in range(ValRepeatClass[i]):
                ImgDestPath = Dir + "/" + Img[:-4] + "_" + str(k) + ".jpg" 
                copyfile(ImgPath, ImgDestPath)
                # print(ImgPath, " to ", ImgDestPath)
            # print("Finished ", Img)
        i += 1
