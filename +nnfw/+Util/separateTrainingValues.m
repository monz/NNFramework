function [ data, indexes ] = separateTrainingValues( input, target, vlFactor, tsFactor )
%SEPARATETRAININGVALUES Separates the input and target values into three chunks of data.
%   The data get separated into training, validating and test data.
%
%   [ data, indexes ] = SEPARATETRAININGVALUES( input, target, vlFactor, tsFactor )
%
%   vlFactor:   validation data factor, e.g. 0.2 corresponds to 20 percent
%               of all data used for validation
%   tsFactor:   test data factor, e.g. 0.05 corresponds to 5 percent of all
%               data used for testing
%
%   All remaining data is used for training.
%
%   Returns
%   data:       cellarray of size 3x2, the first column is for the input
%               values in that order - training, validation, test. The
%               second column is for the target values in the same order.
%   indexes:    cellarray of size 3x1, contains the indexes in the original
%               data vector used for training, validation and test data.
    
    if (vlFactor + tsFactor) >= 100
        error('Separation factors too big. No data for training left.')
    end
    
    % determine the number of elements contained by each data set
    numInputElements = size(input, 2);
    if numInputElements <= 1
        data = cell(3,2);
        data{1,1} = input;
        data{1,2} = target;

        % return indexes
        indexes = cell(3,1);
        indexes{1,1} = 1;
        
        return
    end
    
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
    usedIndexes = containers.Map('KeyType','double','ValueType','double');
    j = 1; tsIndexNum = 1; vlIndexNum = 1;
    while j <= numRandElements
        ind = ceil(mod(rand(1,1)*numInputElements, numInputElements));
        
        indMember = isKey(usedIndexes,ind);
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
        usedIndexes(ind) = ind;
        j = j + 1;
    end
    
    % fill training data set with remaining data - therefore use the unused
    % indizes
    trIndexes = ~ismember(1:size(input,2),cell2mat(values(usedIndexes))); % all UNused indizes
    trInput = input(:, trIndexes);
    trTarget = target(:, trIndexes);
    
    % return the separated training values
    data = cell(3,2);
    data{1,1} = trInput;
    data{1,2} = trTarget;
    data{2,1} = vlInput;
    data{2,2} = vlTarget;
    data{3,1} = tsInput;
    data{3,2} = tsTarget;
    
    % return indexes
    indexes = cell(3,1);
    indexes{1,1} = trIndexes;
    indexes{2,1} = vlIndexes;
    indexes{3,1} = tsIndexes;
end

