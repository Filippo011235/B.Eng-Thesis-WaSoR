clc; clear all; close all;

DatasetDir = cell(1,6);
DatasetDir(:) = {'Dataset_Lean_h0/'};
ClassDir = {'HDPE/', 'LDPE/', 'Other/', 'PET/', 'PP/', 'PS/'};
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

cv = cvpartition(ImgListing,'KFold',noFolds)  % No stratify option: Matlab<2018
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

    figure
    montage(validationSet.Files(1:17:end))
    MontageName = strcat('./Montage/MontageFold', int2str(CurrentFold), '.png'); 
    saveas(gcf, MontageName)

    % Once the bag of features is done the SVM training was
    % performed using the “trainImageCategoryClassifier” function.
    categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

    interval = (CurrentFold-1)*7;

    confMatrixTrain = evaluate(categoryClassifier, trainingSet);
    ScoresMatrix(5:10, 3+interval : 3+interval+5) = confMatrixTrain;
    ScoresMatrix(12, 3+interval) = mean(diag(confMatrixTrain));
    
    confMatrixVal = evaluate(categoryClassifier, validationSet);
    ScoresMatrix(16:21, 3+interval : 3+interval+5) = confMatrixVal;
    ScoresMatrix(23, 3+interval) = mean(diag(confMatrixVal));

    % confMatrixTrain{CurrentFold} = evaluate(categoryClassifier, trainingSet);
    % confMatrixVal{CurrentFold} = evaluate(categoryClassifier, validationSet);
    % mean(diag(confMatrixVal))
end

ScoresMatrix = round(ScoresMatrix, 2)
save('ScoresMatrix.mat','ScoresMatrix')


    % fold1 = cv.test(1);
    % fold2 = cv.test(2);
    % fold3 = cv.test(3);
    % fold4 = cv.test(4);
    % fold5 = cv.test(5);

    % data = [fold1(10:30),fold2(10:30),fold3(10:30),fold4(10:30),fold5(10:30)];
    % h = heatmap(double(data),'ColorbarVisible','off');
    % sorty(h,{'1','2','3','4','5'},'descend')
    % xlabel('Repetition')
    % ylabel('Observation')
    % title('Test Set Observations')
    % saveas(gcf, 'ExampleKFold.png')
