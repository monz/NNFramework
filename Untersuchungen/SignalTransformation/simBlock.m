function [ outData ] = simBlock( nets, data, useToolbox )
%SIMSISO Summary of this function goes here
%   Detailed explanation goes here

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