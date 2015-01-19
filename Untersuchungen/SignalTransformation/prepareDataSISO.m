function [ p, t ] = prepareDataSISO( input, target )
%PREPAREDATASISO Summary of this function goes here
%   Detailed explanation goes here

    numInputs = size(input,1);
    inputDataLength = size(input,2);
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
    end

    % reshape to one row for SISO trainig
    dimension = numInputs*inputDataLength;
    p = reshape(input', 1, dimension);
    t = reshape(t', 1, dimension);

end

