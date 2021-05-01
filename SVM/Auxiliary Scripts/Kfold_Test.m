clc; clear all; close all;

load fisheriris
species = categorical(species);

C = categories(species) % Class names
numClasses = size(C,1);
n = countcats(species) % Number of observations in each class
rng('default') % For reproducibility
cv = cvpartition(species,'KFold',5) 
numFolds = cv.NumTestSets;
nTestData = zeros(numFolds,numClasses);
for i = 1:numFolds
    testClasses = species(cv.test(i));
    nCounts = countcats(testClasses); % Number of test set observations in each class
    nTestData(i,:) = nCounts';
end

bar(nTestData)
xlabel('Test Set (Fold)')
ylabel('Number of Observations')
title('Nonstratified Partition')
legend(C)
saveas(gcf, 'PartitionsGraph.png')