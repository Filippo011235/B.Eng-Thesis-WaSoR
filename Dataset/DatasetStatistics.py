from os import listdir, scandir

ClassesLabels = ["PET", "HDPE", "LDPE", "PP", "PS", "Other"]
# Get dir names and print them, just to be sure ;) 
SubDirList = [f.path for f in scandir('./') if f.is_dir()]
SubDirList = SubDirList[1:7] # ./a01 <-> a07
print(SubDirList)

ClassImages2DArr = [] # 2D array storing listdir from each class dir
ClassIndices = []
ObjectsIndices = []
ImgClassQty = []
ObjectClassQty = []
ImgPerObjectClassMean = []
MineIndices = 0
PercentMineClass = []

ImgQty, ObjectQty = 0, 0


for Dir in SubDirList:
    ClassImages2DArr.append(listdir(Dir))

for ClassList in ClassImages2DArr:
    # Qauntity values
    ImgClassQty.append(len(ClassList)) # Total no. of Img in a dir
    for Img in ClassList:
        ObjectIdx = int(Img[:4]) # Each ImgName starts with object idx number, e.g. "0321**"
        if ObjectIdx not in ClassIndices:
            ClassIndices.append(ObjectIdx)   # collect each new object Idx
            if ObjectIdx > 100:
                MineIndices += 1
    ObjectsIndices.append(ClassIndices)
    ObjectClassQty.append(len(ObjectsIndices[-1]))

    # Percent mine
    Kokosik = MineIndices/len(ClassIndices)
    PercentMineClass.append(Kokosik)
    MineIndices = 0  # Start anew for each class
    ClassIndices.clear()        # Start anew for each class

    # Mean value
    Koko = ImgClassQty[-1]/ObjectClassQty[-1]
    ImgPerObjectClassMean.append(Koko)


for Qty in ImgClassQty:
    ImgQty += Qty # sum of all

for Qty in ObjectClassQty:
    ObjectQty += Qty # sum of all

ImgObjectMean = ImgQty/ObjectQty

print("----------------------------------------\n")
print("Quantity: ")
print("Images qty: ", ImgQty)
print("Object qty: ", ObjectQty)
print("ClassesLabels:")
for i in range(len(ImgClassQty)):
    print(ClassesLabels[i], ": Img = ", ImgClassQty[i], ";   Objects = ", ObjectClassQty[i])

print(ImgQty)
print(ObjectQty)
for i in range(len(ImgClassQty)):
    print(ImgClassQty[i])
print("\n")
for i in range(len(ImgClassQty)):
    print(ObjectClassQty[i])

print("----------------------------------------\n")
print("Mean: ")
print("Images per Object: ", round(ImgObjectMean,1))
print("ClassesLabels:")
for i in range(len(ImgPerObjectClassMean)):
    print(ClassesLabels[i], ": Img per Object = ", round(ImgPerObjectClassMean[i],1))

print(round(ImgObjectMean,1))
for i in range(len(ImgPerObjectClassMean)):
    print(round(ImgPerObjectClassMean[i],1))


print("----------------------------------------\n")
print("Percentage of mine idx per class: ")
for i in range(len(PercentMineClass)):
    print(ClassesLabels[i], ": Percentage = ", round(PercentMineClass[i] * 100,1), "%")
for i in range(len(PercentMineClass)):
    print(round(PercentMineClass[i] * 100,1))
