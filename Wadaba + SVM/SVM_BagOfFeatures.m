%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code has been basically ripped off from:
% https://www.mathworks.com/help/vision/ug/image-category-classification-using-bag-of-features.html
% Comments from G. E. Sakr et. al. paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;

% Load & prepare the dataset
imds = imageDatastore('Wadaba_Main','IncludeSubfolders',true,'LabelSource','foldernames');
% unzip('MerchData.zip');
% imds = imageDatastore('MerchData','IncludeSubfolders',true,'LabelSource','foldernames');

tbl = countEachLabel(imds);
figure
montage(imds.Files(1:16:end))
saveas(gcf, 'Montage.png')
% savefig('Montage.fig')

[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomize');

%%
% In order to get the bag of features needed to train
% SVM, the “bagOfFeatures” function was used.
% This method takes the training set as a parameter as well as
% the number of clusters for K-means. [...] 
% The K that yielded that highest accuracy is 500 and was used on the
% test set. 
bag = bagOfFeatures(trainingSet);

img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')
saveas(gcf, 'Histogram.png')

% Once the bag of features is done the SVM training was
% performed using the “trainImageCategoryClassifier” function.
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
confMatrix = evaluate(categoryClassifier, trainingSet);
confMatrix = evaluate(categoryClassifier, validationSet);
mean(diag(confMatrix)) % Compute average accuracy

% This function returns an SVM model that can be applied to the
% test set by using the function “evaluate”.

% img = imread(fullfile('MerchData','MathWorks Cap','Hat_0.jpg'));
% [labelIdx, scores] = predict(categoryClassifier, img);
% categoryClassifier.Labels(labelIdx)
