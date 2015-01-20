function [ outData ] = simSISO( net, data, useToolbox )
%SIMSISO Summary of this function goes here
%   Detailed explanation goes here

    numTests = data.numTests;
    dataSize = data.size;
    targetSize = length(data.t);
    yTest = zeros(1, numTests);
    fitTest = zeros(1, numTests);
    dbTest = zeros(1, numTests);
    
    if useToolbox
        y = net(data.p);
        yExtra = net(data.extraP);
        for k = 1:numTests
            ind = (k-1)*dataSize+1:k*dataSize;
            indEndTarget = min(targetSize, k*dataSize);
            indStartTarget = (indEndTarget - dataSize) +1;
            yTest(ind) = net(data.testP(ind));
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', data.t(indStartTarget:indEndTarget)', 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), data.t(indStartTarget:indEndTarget));
        end
    else
        y = net.simulate(data.p);
        yExtra = net.simulate(data.extraP);
        for k = 1:numTests
            ind = (k-1)*dataSize+1:k*dataSize;
            indEndTarget = min(targetSize, k*length(ind));
            indStartTarget = (indEndTarget - dataSize) +1;
            yTest(ind) = net.simulate(data.testP(ind));
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', data.t(indStartTarget:indEndTarget)', 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), data.t(indStartTarget:indEndTarget));
        end
    end

    outData.fit = nnfw.goodnessOfFit(y', data.t', 'NRMSE');
    outData.fitExtra = nnfw.goodnessOfFit(yExtra', data.t(1:dataSize)', 'NRMSE');
    outData.db = daviesBouldin(y, data.t);
    outData.dbExtra = daviesBouldin(yExtra, data.t(1:dataSize));
    
    outData.y = y;
    outData.yTest = yTest;
    outData.yExtra = yExtra;
    outData.fitTest = fitTest;
    outData.dbTest = dbTest;

end