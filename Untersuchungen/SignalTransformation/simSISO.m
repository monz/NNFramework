function [ outData ] = simSISO( net, data, useToolbox )
%SIMSISO Summary of this function goes here
%   Detailed explanation goes here

    numTests = data.numTests;
    dataSize = data.size;
    yTest = zeros(1, numTests);
    fitTest = zeros(1, numTests);
    dbTest = zeros(1, numTests);
    
    if useToolbox
        y = net(data.p);
        yExtra = net(data.extraP);
        for k = 1:numTests
            ind = (k-1)*dataSize+1:k*dataSize;
            yTest(ind) = net(data.testP(ind));
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', data.t(ind)', 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), data.t(ind));
        end
    else
        y = net.simulate(data.p);
        yExtra = net.simulate(data.extraP);
        for k = 1:numTests
            ind = (k-1)*dataSize+1:k*dataSize;
            yTest(ind) = net.simulate(data.testP(ind));
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', data.t(ind)', 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), data.t(ind));
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