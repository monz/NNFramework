function [ output ] = separateTrainingValues( input, target, vlFactor, tsFactor )
%SEPARATEINPUTVALUES Summary of this function goes here
%   Detailed explanation goes here

%     vlFactor = validation data factor
%     tsFactor = test data factor
    
    if (vlFactor + tsFactor) >= 100
        error('Separation factors too big. No data for training left.')
    end
    
    % determine the number of elements containt by each data set
    numInputElements = length(input);
    numVlElements = floor(numInputElements * vlFactor);
    numTsElements = ceil(numInputElements * tsFactor);
    numTrElements = numInputElements - numVlElements - numTsElements;

    trInput = NaN(size(input,1),numTrElements); % contains e.g. 75-percent of data
    vlInput = NaN(size(input,1),numVlElements); % 20-percent
    tsInput = NaN(size(input,1),numTsElements); % 5-percent
        
    trTarget = NaN(size(target,1),numTrElements);
    vlTarget = NaN(size(target,1),numVlElements);
    tsTarget = NaN(size(target,1),numTsElements);
   
    % fill validation and test data sets with randomly choosen data.
    numRandElements = numVlElements + numTsElements;
    usedIndizes = zeros(1, numRandElements);
    j = 1; tsIndex = 1; vlIndex = 1;
    while j <= numRandElements
        ind = ceil(mod(rand(1,1)*numInputElements, numInputElements));
        
        if ~ismember(ind, vlInput) && vlIndex <= numVlElements
            vlInput(:, vlIndex) = input(:, ind);
            vlTarget(:, vlIndex) = target(:, ind);
            
            vlIndex = vlIndex + 1;
%             j = j + 1;
        elseif ~ismember(ind, tsInput) && tsIndex <= numTsElements
            tsInput(:, tsIndex) = input(:, ind);
            tsTarget(:, tsIndex) = target(:, ind);
            
            tsIndex = tsIndex + 1;
%             j = j + 1;
        else
           continue 
        end
        usedIndizes(j) = ind;
        j = j + 1;
    end
    
    % fill training data set with remaining data - therefor use the unused
    % indizes
    trIndex = 1;
    for ind = 1:numInputElements
       if ~ismember(ind, usedIndizes) && trIndex <= numTrElements
          trInput(:, trIndex) = input(:, ind);
          trTarget(:, trIndex) = target(:, ind);
          
          trIndex = trIndex + 1;
       end
    end
    
    % return the separated training values
    output = cell(3,2);
    output{1,1} = trInput;
    output{1,2} = trTarget;
    output{2,1} = vlInput;
    output{2,2} = vlTarget;
    output{3,1} = tsInput;
    output{3,2} = tsTarget;
end

