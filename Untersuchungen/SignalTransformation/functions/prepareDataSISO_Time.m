function [ p, t, teData, exData ] = prepareDataSISO_Time( input, target, testData, extraData, times )
%PREPAREDATASISO Prepares the data for SISO_Time training method
%   Each training method needs specific preprocessing of the training data.
%   Therefore exists different prepare functions which handle the data
%   specific to their usage in the neural network.

    % calculate sizes
    numInputs = size(input,1);
    inputDataLength = size(input,2);
    numTargets = size(target,1);
    numTests = size(testData,1);
    numExtra = size(extraData,1);
    timesDataLength = size(times,2);
    
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

    % reshape to one row for SISO trainig
    dimensionInput = numInputs*inputDataLength;
    dimensionTests = numTests*inputDataLength;
    dimensionExtra = numExtra*inputDataLength;
    inputData = reshape(input', 1, dimensionInput);
    targetData = reshape(t', 1, dimensionInput);
    teDataWT = reshape(testData', 1, dimensionTests);
    exDataWT = reshape(extraData', 1, dimensionExtra);

    % prepare time rows for all input data
    timeRow = reshape(times', 1, timesDataLength);
    timeInput = repmat(timeRow, 1, numInputs);
    timeTarget = repmat(timeRow, 1, numInputs);
    timeTests = repmat(timeRow, 1, numTests);
    timeExtra = repmat(timeRow, 1, numExtra);
    
    % combine input data with time data
    p = [inputData; timeInput];
    t = [targetData; timeTarget];
    teData = [teDataWT; timeTests];
    exData = [exDataWT; timeExtra];
end

