from os import listdir

TrainIdx = []
TestIdx = []

TrainImgList = listdir("./PostProcessing Data/Training/")
for Img in TrainImgList:
    Idx = Img[1:4]
    if Idx not in TrainIdx:
        TrainIdx.append(Idx)
TrainIdx.sort()

TestImgList = listdir("./PostProcessing Data/Test/")
for Img in TestImgList:
    Idx = Img[1:4]
    if Idx not in TestIdx:
        TestIdx.append(Idx)
TestIdx.sort()

print("Test: ")
print(TestIdx)
print("\n ------------------------------- \n")
print("Train: ")
print(TrainIdx)

for Idx in TestIdx:
    if Idx in TrainIdx:
        print(Idx) # these appear in both sets
print("Finished checking")