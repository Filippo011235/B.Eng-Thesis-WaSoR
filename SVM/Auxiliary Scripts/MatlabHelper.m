PetSet = imageDatastore('./Dataset_Opti/PET_bot','LabelSource','foldernames');
LDPEIdx = [];
MiscIdx = [];

j = 1;
k = 1;
for i = 1:365
    I = readimage(PetSet, i);
    LabelIdx = predict(categoryClassifier, I);
    if LabelIdx < 4 && LabelIdx > 1
        if LabelIdx == 2
            LDPEIdx(j) = i;
            j = j +1;
        end
        if LabelIdx == 3
            MiscIdx(k) = i;
            k = k +1;
        end
    end
end
for imgNo = 1:length(LDPEIdx)
    I = readimage(PetSet, LDPEIdx(imgNo));
    [featureVector,words] = encode(bag, I);
    pts = double(words.Location);
    figure, imshow(I);
    hold on
    plot(pts(:,1), pts(:,2), 'gs', 'MarkerSize', 10);
    fig_path = strcat('./Pet_bot_analysis/LDPE_', int2str(imgNo), '.png');
    saveas(gcf, fig_path);
end

for imgNo = 1:length(MiscIdx)
    I = readimage(PetSet, MiscIdx(imgNo));
    [featureVector,words] = encode(bag, I);
    pts = double(words.Location);
    figure, imshow(I);
    hold on
    plot(pts(:,1), pts(:,2), 'gs', 'MarkerSize', 10);
    fig_path = strcat('./Pet_bot_analysis/Misc_', int2str(imgNo), '.png');
    saveas(gcf, fig_path);
end
