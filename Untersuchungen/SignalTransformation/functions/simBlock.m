function [ outData ] = simBlock( nets, data, useToolbox )
%SIMBLOCK Simulates the data prepared for Block training method
%   Each training method preprocesses the training data differently.
%   Therefore exists different simulation functions which handle the data
%   specific to their preparation.
%
%   nets:           cellarray of all trained neural networks - each chunk of
%                   the signal gets its own network trained
%
%   data is of type struct and should contain at least following
%   information:
%
%   numTests:        number of test singals
%   size:           data size of the input signals
%   numNets:        number of networks to train, depends on maxDimension
%                   and data size
%   maxDimension:   maximum size of neural network input dimension
%   numInputs:      number of input signals
%   p:              net input signals
%   testP:          net test input signals
%   extraP:         net extrapolation input signals
%   t:              target data the neural network should "learn"
%
%   useToolbox:     if true uses the toolbox to simulate the data,
%                   otherwise it uses the NN-Framework

    numTests = data.numTests;
    dataSize = data.size;
    targetSize = data.size;
    y = zeros(dataSize,1);
    yTest = zeros(dataSize, numTests);
    fitTest = zeros(1, numTests);
    dbTest = zeros(1, numTests);
    numNets = data.numNets;
    maxDimension = data.maxDimension;
    
    if useToolbox
        
        missingValues = max(dataSize, dataSize-maxDimension);
        for l = 1:numNets
            startInd = (l-1) * maxDimension + 1;
            endInd = min((startInd - 1) + missingValues, l*maxDimension);
            missingValues = missingValues - min(maxDimension, missingValues);

            y(startInd:endInd) = nets{l}(data.p(startInd:endInd));
            yExtra(startInd:endInd) = nets{l}(data.extraP(startInd:endInd));
        end
        
        for k = 1:numTests
            
            missingValues = max(dataSize, dataSize-maxDimension);
            for l = 1:numNets
                startInd = (l-1) * maxDimension + 1;
                endInd = min((startInd - 1) + missingValues, l*maxDimension);
                missingValues = missingValues - min(maxDimension, missingValues);

                yTest(startInd:endInd,k) = nets{l}(data.testP(startInd:endInd,k));
            end
            
            ind = (k-1)*dataSize+1:k*dataSize;
            indEndTarget = min(targetSize, k*dataSize);
            indStartTarget = (indEndTarget - dataSize) +1;
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', data.t(indStartTarget:indEndTarget), 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), data.t(indStartTarget:indEndTarget));
        end
    else
        missingValues = max(dataSize, dataSize-maxDimension);
        for l = 1:numNets
            startInd = (l-1) * maxDimension + 1;
            endInd = min((startInd - 1) + missingValues, l*maxDimension);
            missingValues = missingValues - min(maxDimension, missingValues);

            y(startInd:endInd) = nets{l}.simulate(data.p(startInd:endInd));
            yExtra(startInd:endInd) = nets{l}.simulate(data.extraP(startInd:endInd));
        end
        
        for k = 1:numTests
            
            missingValues = max(dataSize, dataSize-maxDimension);
            for l = 1:numNets
                startInd = (l-1) * maxDimension + 1;
                endInd = min((startInd - 1) + missingValues, l*maxDimension);
                missingValues = missingValues - min(maxDimension, missingValues);

                yTest(startInd:endInd,k) = nets{l}.simulate(data.testP(startInd:endInd,k));
            end
            
            ind = (k-1)*dataSize+1:k*dataSize;
            indEndTarget = min(targetSize, k*dataSize);
            indStartTarget = (indEndTarget - dataSize) +1;
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', data.t(indStartTarget:indEndTarget), 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), data.t(indStartTarget:indEndTarget));
        end
    end

    outData.fit = nnfw.goodnessOfFit(y, data.t, 'NRMSE');
    outData.fitExtra = nnfw.goodnessOfFit(yExtra', data.t(1:dataSize), 'NRMSE');
    outData.db = daviesBouldin(y, data.t);
    outData.dbExtra = daviesBouldin(yExtra, data.t(1:dataSize));
    
    outData.y = y;
    outData.yTest = yTest;
    outData.yExtra = yExtra;
    outData.fitTest = fitTest;
    outData.dbTest = dbTest;

end