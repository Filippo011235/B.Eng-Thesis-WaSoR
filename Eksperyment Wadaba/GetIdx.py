from os import listdir

ImgNames = listdir("./Raw from Wadaba")
ImgNames.sort() # Ordnung muss sein

f = open("./WadabaIdx.txt", 'w')

IdxArr = [] # stores unique idx

for Img in ImgNames:
    WasteIdx = Img[0:4]     # get idx
    if WasteIdx not in IdxArr:      # is it first time here?
        WasteClass = Img[7]     # get class
        WasteInfo = WasteIdx + "    " + WasteClass + "\n"
        IdxArr.append(WasteIdx)
        f.write(WasteInfo)
f.close()

        