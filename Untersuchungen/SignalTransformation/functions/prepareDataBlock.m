function [ p, t, teData, exData ] = prepareDataBlock( input, target, testData, extraData )
%PREPAREDATASISO Prepares the data for Block training method
%   Each training method needs specific preprocessing of the training data.
%   Therefore exists different prepare functions which handle the data
%   specific to their usage in the neural network.

    numInputs = size(input,1);
    numTargets = size(target,1);
    % repead target values until numInputs and numTargets are equal
    t = zeros(numInputs,length(target));
    if numTargets < numInputs
        t(1:numTargets,:) = target(1:numTargets,:);
        offset = numTargets;
        missingValues = numInputs-offset;
        while(missingValues > 0)
            ind = min(numTargets,missingValues);
            t(offset+1:offset+ind,:) = target(1:ind,:);
            offset = offset + ind;
            missingValues = numInputs-offset;
        end
    elseif numTargets > numInputs
        t = target(1:numInputs,:);
    else
        t = target;
    end
    
    p = input';
    t  = t';
    teData = testData';
    exData = extraData';
end

