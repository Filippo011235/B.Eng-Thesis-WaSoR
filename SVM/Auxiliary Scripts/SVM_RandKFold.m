% SVM with basic random KFold, no stratification

clc; clear all; close all;

MatrixSavePath = './ScoreMatrices/All_0016n.mat';
% MatrixSavePath = './ScoreMatrices/All_Grid_0016y.mat';
DatasetPath = './Dataset_Original_0016n/';
noFolds = 5;

ClassDir = {'HDPE/', 'LDPE/', 'Other/', 'PET/', 'PP/', 'PS/'};
noClass = size(ClassDir,2);
DatasetDir = cell(1,noClass);
DatasetDir(:) = {DatasetPath};
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

ScoresMatrix = zeros(24, (noFolds-1)*7+3+1+5);
% ListingOfValListing = cell(noFolds,1);

% rng('default') % For reproducibility
rng('shuffle')

IL = size(ImgListing, 2);
cv = cvpartition(IL,'KFold',noFolds)
% cv = cvpartition(ImgListing,'KFold',noFolds)

% Temporar storage for idx of current fold train. imgs:
TrainingFold = false(1, size(cv.training(1),1)); 

% Iterate over Folds
for CurrentFold = 1:noFolds 
    TrainingFold(:) = cv.training(CurrentFold);

    trainingListing = [];
    validationListing = [];

    for i = 1:size(ImgListing,2)
        if TrainingFold(i)
            trainingListing = [trainingListing, ImgListing(i)];
        else 
            validationListing = [validationListing, ImgListing(i)];
        end
    end

    % Load & prepare the dataset
    trainingSet = imageDatastore(trainingListing,'LabelSource','foldernames');
    validationSet = imageDatastore(validationListing,'LabelSource','foldernames');

    %%
    % In order to get the bag of features needed to train
    % SVM, the “bagOfFeatures” function was used.
    % This method takes the training set as a parameter as well as
    % the number of clusters for K-means. [...] 
    % The K that yielded that highest accuracy is 500 and was used on the
    % test set. 
    
    bag = bagOfFeatures(trainingSet, 'PointSelection', 'Detector');
    % bag = bagOfFeatures(trainingSet);

    figure;
    montage(validationSet.Files(1:18:end));
    MontageName = strcat('./Montage/MontageFold', int2str(CurrentFold), '.png'); 
    saveas(gcf, MontageName);

    % Once the bag of features is done the SVM training was
    % performed using the “trainImageCategoryClassifier” function.
    categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

    interval = (CurrentFold-1)*7;

    confMatrixTrain = evaluate(categoryClassifier, trainingSet);
    ScoresMatrix(5:5+noClass-1, 3+interval : 3+interval+noClass-1) = confMatrixTrain;
    ScoresMatrix(12, 3+interval) = mean(diag(confMatrixTrain));
    
    confMatrixVal = evaluate(categoryClassifier, validationSet);
    ScoresMatrix(16:16+noClass-1, 3+interval : 3+interval+noClass-1) = confMatrixVal;
    ScoresMatrix(23, 3+interval) = mean(diag(confMatrixVal));

    % confMatrixTrain{CurrentFold} = evaluate(categoryClassifier, trainingSet);
    % confMatrixVal{CurrentFold} = evaluate(categoryClassifier, validationSet);
    % mean(diag(confMatrixVal))
end

% save('./PoC/LoL.mat', 'ListingOfValListing');

ScoresMatrix = round(ScoresMatrix, 2);
save(MatrixSavePath,'ScoresMatrix');

