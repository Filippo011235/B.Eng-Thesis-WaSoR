% - SVM with bagOfFeatures and stratified KFold. 
% - Due to my Matlab version, I had to make regular KFolds within classes
% then mix them for stratification.

% Matlab setup:
clc; clear all; close all;

% Datasets and results paths.
ExperimentsPath = {'./Dataset_Final/', './Dataset_Opti/'};
ResultsMatricesPath = {'./ScoreMatrices/F_Final_D_rng3.mat', './ScoreMatrices/F_Opti_D_K10.mat'};

% Experiment setup
TypeOfExperiment = 2; % which dataset you wanna use?
MatrixSavePath = ResultsMatricesPath{TypeOfExperiment};
DatasetPath = ExperimentsPath{TypeOfExperiment};
BagUsingGrid = 0;   % which parameter of bagOfFeatures? 1 - Grid; 0 - Detector
noFolds = 10;       % K in KFolds
rng(0)              % For reproducibility

% ClassDir = {'HDPE/', 'LDPE/', 'Other/', 'PET/', 'PP/', 'PS/'}; % All class
ClassDir = {'HDPE/', 'LDPE/', 'Misc/', 'PET_bot/'};  % Opti

% KFold setup
noClass = size(ClassDir,2);
DatasetDir = cell(1,noClass);
DatasetDir(:) = {DatasetPath};
ScrPath = strcat(DatasetDir, ClassDir);
ClassListing = cell(1,noClass);
ClassSize = cell(1,noClass);
ClassCVPart = cell(1,noClass);

% Create Kfold object for each class
for Class = 1:noClass
    ImgsStruct = dir(ScrPath{Class});
    DirPath = cell(1,size(ImgsStruct,1)-2);
    DirPath(:) = {ScrPath{Class}};
    ImgNames = {ImgsStruct.name};
    ImgNames = ImgNames(3:end); % remove first two elements("/.", "/..")
    ImgPathNames = strcat(DirPath, ImgNames);

    ClassListing{Class} = ImgPathNames;
    ClassSize{Class} = length(ImgPathNames);
    % Regular KFold within Class
    ClassCVPart{Class} = cvpartition(length(ImgPathNames), 'KFold', noFolds);
end

% Matrix to store results for each fold
ScoresMatrix = zeros(24, (noFolds-1)*7+3+1+5);

% IL = size(ImgListing, 2);
% cv = cvpartition(IL,'KFold',noFolds)
% cv = cvpartition(ImgListing,'KFold',noFolds)

% Temporar storage for idx of current fold train. imgs:

% Montage dir stores example images from experiment; 
% Useful for... getting in touch with dataset, noticing abnormalities.
delete('./Montage/*'); % Delete previous Montage files

% From KFold objects do actual train., valid. listing of images
for CurrentFold = 1:noFolds 
    trainingListing = [];
    validationListing = [];

    for Class = 1:noClass
        CurrentClassCVPart = ClassCVPart{Class}; % Recall Kfold for given class
        TrainingFold = false(1, ClassSize{Class}); % vector of zeros
        TrainingFold(:) = CurrentClassCVPart.training(CurrentFold); % overwrite
        
        for i = 1:ClassSize{Class} % For each image in the Class
            if TrainingFold(i)  % If designated as Training, by CurrentClassCVPart:
                trainingListing = [trainingListing, ClassListing{Class}(i)];
            else % If Validation:
                validationListing = [validationListing, ClassListing{Class}(i)];
            end
        end
    end

    % Randomize listings to fully emulate stratification
    trainingListing = trainingListing(randperm(length(trainingListing)));
    validationListing = validationListing(randperm(length(validationListing)));
    
    % Load & prepare the dataset, based on listings
    trainingSet = imageDatastore(trainingListing,'LabelSource','foldernames');
    validationSet = imageDatastore(validationListing,'LabelSource','foldernames');

    % From Sakr et. al. paper:
    % In order to get the bag of features needed to train
    % SVM, the “bagOfFeatures” function was used.
    % This method takes the training set as a parameter as well as
    % the number of clusters for K-means. [...] 
    % The K that yielded that highest accuracy is 500 and was used on the
    % test set. 
    if BagUsingGrid     % Grid vs Detector, declared earlier
        bag = bagOfFeatures(trainingSet);
    else
        bag = bagOfFeatures(trainingSet, 'PointSelection', 'Detector');
    end

    % Once the bag of features is done the SVM training was
    % performed using the “trainImageCategoryClassifier” function.
    categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

    % Save results, confusion matrices, as a large "ScoreMatrix"
    interval = (CurrentFold-1)*7; % Number of folds matrices

    % save Training dataset results
    confMatrixTrain = evaluate(categoryClassifier, trainingSet);
    ScoresMatrix(5:5+noClass-1, 3+interval : 3+interval+noClass-1) = confMatrixTrain;
    ScoresMatrix(12, 3+interval) = mean(diag(confMatrixTrain));
    
    % save Validation dataset results
    confMatrixVal = evaluate(categoryClassifier, validationSet);
    ScoresMatrix(16:16+noClass-1, 3+interval : 3+interval+noClass-1) = confMatrixVal;
    ScoresMatrix(23, 3+interval) = mean(diag(confMatrixVal));

    figure; % Montage of some example images
    montage(validationSet.Files(1:18:end));
    MontageName = strcat('./Montage/MontageFold', int2str(CurrentFold), '.png'); 
    saveas(gcf, MontageName);
end

% save('./PoC/LoL.mat', 'ListingOfValListing');

ScoresMatrix = round(ScoresMatrix, 2); % For readability
save(MatrixSavePath,'ScoresMatrix'); % Save the Matrix and do with it what you wish

