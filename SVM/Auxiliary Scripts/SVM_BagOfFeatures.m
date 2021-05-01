%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Most of code comes from:
% https://www.mathworks.com/help/vision/ug/image-category-classification-using-bag-of-features.html
% Comments mostly from G. E. Sakr et. al. paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;

% Load & prepare the dataset
imds = imageDatastore('Dataset_Lean_h0','IncludeSubfolders',true,'LabelSource','foldernames');

% Some example data visualization
% figure
% montage(imds.Files(1:20:end))
% saveas(gcf, 'Montage.png')

[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomize');

%%
% In order to get the bag of features needed to train
% SVM, the “bagOfFeatures” function was used.
% This method takes the training set as a parameter as well as
% the number of clusters for K-means. [...] 
% The K that yielded that highest accuracy is 500 and was used on the
% test set. 
bag = bagOfFeatures(trainingSet, 'PointSelection', 'Detector');
% bag2 = bagOfFeatures(trainingSet);
% Show locations of visual words.

% % figure, imshow(I);
% % for i=1:words.Count
% %     text(pts(i,1), pts(i,2), sprintf('%d', words.WordIndex(i)), ...
% %     'Color', 'g', 'FontSize', 8);
% % end


% I = readimage(validationSet, 2);
% [featureVector,words] = encode(bag, I);
% pts = double(words.Location);
% figure, imshow(I);
% hold on
% plot(pts(:,1), pts(:,2), 'g.');
% figure, imshow(I);
% for i=1:words.Count
%     text(pts(i,1), pts(i,2), sprintf('%d', words.WordIndex(i)), ...
%     'Color', 'g', 'FontSize', 8);
% end
% saveas(gcf, 'WordRepresent2.png')

% I = readimage(validationSet, 3);
% [featureVector,words] = encode(bag, I);
% pts = double(words.Location);
% figure, imshow(I);
% hold on
% plot(pts(:,1), pts(:,2), 'g.');
% figure, imshow(I);
% for i=1:words.Count
%     text(pts(i,1), pts(i,2), sprintf('%d', words.WordIndex(i)), ...
%     'Color', 'g', 'FontSize', 8);
% end
% saveas(gcf, 'WordRepresent3.png')

% I = readimage(validationSet, 4);
% [featureVector,words] = encode(bag, I);
% pts = double(words.Location);
% figure, imshow(I);
% hold on
% plot(pts(:,1), pts(:,2), 'g.');
% figure, imshow(I);
% for i=1:words.Count
%     text(pts(i,1), pts(i,2), sprintf('%d', words.WordIndex(i)), ...
%     'Color', 'g', 'FontSize', 8);
% end
% saveas(gcf, 'WordRepresent4.png')

% Plot the histogram of visual word occurrences
% img = readimage(imds, 1);
% featureVector = encode(bag, img);
% figure
% bar(featureVector)
% title('Visual word occurrences')
% xlabel('Visual word index')
% ylabel('Frequency of occurrence')
% saveas(gcf, 'Histogram.png')

% Once the bag of features is done the SVM training was
% performed using the “trainImageCategoryClassifier” function.
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

% confMatrix = evaluate(categoryClassifier, trainingSet);
% confMatrix = evaluate(categoryClassifier, validationSet);


% This function returns an SVM model that can be applied to the
% test set by using the function “evaluate”.
% img = imread(fullfile('MerchData','MathWorks Cap','Hat_0.jpg'));
% [labelIdx, scores] = predict(categoryClassifier, img);
% categoryClassifier.Labels(labelIdx)