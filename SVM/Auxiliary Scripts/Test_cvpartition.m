clc; clear all; close all;

ClassDir = {'HDPE/', 'LDPE/', 'Other/', 'PET/', 'PP/', 'PS/'};
DatasetDir = cell(1, size(ClassDir,2));
DatasetDir(:) = {'Dataset_Lean_h0/'};

ScrPath = strcat(DatasetDir, ClassDir);
ImgListing = [];

for i = 1:size(ScrPath,2)
    ImgsStruct = dir(ScrPath{i});
    DirPath = cell(1,size(ImgsStruct,1)-2);
    DirPath(:) = {ScrPath{i}};
    ImgNames = {ImgsStruct.name};
    ImgNames = ImgNames(3:end); % remove first two elements("/.", "/..")
    ImgPathNames = strcat(DirPath, ImgNames);
    ImgListing = [ImgListing, ImgPathNames];
end

% ImgListing = ImgListing.'

noFolds = 5;
ScoresMatrix = zeros(24, (noFolds-1)*7+3+1+5);
% confMatrixTrain = cell(1, noFolds);
% TrainAcc = cell(1, noFolds);
% confMatrixVal = cell(1, noFolds);
% ValAcc = cell(1, noFolds);

cv = cvpartition(ImgListing,'KFold',noFolds)
% Temporar storage for idx of current fold train. imgs:
TrainingFold = false(1, size(cv.training(1),1)); 

ImgListing = categorical(ImgListing);

C = categories(ImgListing) % Class names
numClasses = size(C,1);
n = countcats(ImgListing) % Number of observations in each class
rng('default') % For reproducibility
cv = cvpartition(ImgListing,'KFold',5) 
numFolds = cv.NumTestSets;
nTestData = zeros(numFolds,numClasses);
for i = 1:numFolds
    testClasses = ImgListing(cv.test(i));
    nCounts = countcats(testClasses); % Number of test set observations in each class
    nTestData(i,:) = nCounts';
end

bar(nTestData)
xlabel('Test Set (Fold)')
ylabel('Number of Observations')
title('Nonstratified Partition')
legend(C)
saveas(gcf, 'PartitionsGraph.png')