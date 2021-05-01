% Script for visualizing given image as an histogram based on bagOfFeatures

DatasetDir = 'Dataset_h0y';
VisWordName = './VisWords/Words_X.png';
ImgNo = 106;
% Load & prepare the dataset
imds = imageDatastore(DatasetDir,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomize');
bag = bagOfFeatures(trainingSet, 'PointSelection', 'Detector');
bag2 = bagOfFeatures(trainingSet);

% Show locations of visual words.
I = readimage(imds, ImgNo);
[featureVector,words] = encode(bag, I);
pts = double(words.Location);
figure, imshow(I);
hold on
plot(pts(:,1), pts(:,2), 'gs', 'MarkerSize', 10);
saveas(gcf, './VisWords/W_LDPE_100_D.png');

[featureVector,words] = encode(bag2, I);
pts = double(words.Location);
figure, imshow(I);
hold on
plot(pts(:,1), pts(:,2), 'gs', 'MarkerSize', 10);
saveas(gcf, './VisWords/W_LDPE_100_G.png');
