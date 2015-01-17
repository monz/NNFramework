function [ values, indexes ] = separateTrainingValues( input, target, vlFactor, tsFactor )
%SEPARATEINPUTVALUES Summary of this function goes here
%   Detailed explanation goes here

%     vlFactor = validation data factor
%     tsFactor = test data factor
    
    if (vlFactor + tsFactor) >= 100
        error('Separation factors too big. No data for training left.')
    end
    
    % determine the number of elements contained by each data set
    numInputElements = size(input, 2);
    numVlElements = floor(numInputElements * vlFactor);
    numTsElements = ceil(numInputElements * tsFactor);

    % training data contains e.g. 75-percent of data
    vlInput = NaN(size(input,1),numVlElements); % validation data 20-percent
    tsInput = NaN(size(input,1),numTsElements); % test data 5-percent
        
    vlTarget = NaN(size(target,1),numVlElements);
    tsTarget = NaN(size(target,1),numTsElements);
    
    vlIndexes = zeros(1, numVlElements);
    tsIndexes = zeros(1, numTsElements);
   
    % fill validation and test data sets with randomly choosen data.
    numRandElements = numVlElements + numTsElements;
    usedIndizes = zeros(1, numRandElements);
    j = 1; tsIndexNum = 1; vlIndexNum = 1;
    while j <= numRandElements
        ind = ceil(mod(rand(1,1)*numInputElements, numInputElements));
        
        indMember = ismember(ind, usedIndizes);
        if ~indMember && vlIndexNum <= numVlElements
            vlInput(:, vlIndexNum) = input(:, ind);
            vlTarget(:, vlIndexNum) = target(:, ind);
            
            % save index
            vlIndexes(vlIndexNum) = ind;
            
            vlIndexNum = vlIndexNum + 1;
        elseif ~indMember && tsIndexNum <= numTsElements
            tsInput(:, tsIndexNum) = input(:, ind);
            tsTarget(:, tsIndexNum) = target(:, ind);
            
            % save index
            tsIndexes(tsIndexNum) = ind;
            
            tsIndexNum = tsIndexNum + 1;
        else
           continue 
        end
        usedIndizes(j) = ind;
        j = j + 1;
    end
    
    % fill training data set with remaining data - therefore use the unused
    % indizes
    trIndexes = ~ismember(1:size(input,2),usedIndizes); % all UNused indizes
    trInput = input(:, trIndexes);
    trTarget = target(:, trIndexes);
    
    % return the separated training values
    values = cell(3,2);
    values{1,1} = trInput;
    values{1,2} = trTarget;
    values{2,1} = vlInput;
    values{2,2} = vlTarget;
    values{3,1} = tsInput;
    values{3,2} = tsTarget;
    
    % return indexes
    indexes = cell(3,1);
    indexes{1,1} = trIndexes;
    indexes{2,1} = vlIndexes;
    indexes{3,1} = tsIndexes;
end

