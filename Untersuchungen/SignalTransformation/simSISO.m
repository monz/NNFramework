function [ y, yTest, fit, fitTest, db, dbTest ] = simSISO( net, data, useToolbox )
%SIMSISO Summary of this function goes here
%   Detailed explanation goes here

    numTests = data.numTests;
    dataSize = data.size;
    yTest = zeros(1, numTests);
    fitTest = zeros(1, numTests);
    dbTest = zeros(1, numTests);
    
    if useToolbox
        y = net(data.p);
        for k = 1:numTests
            ind = (k-1)*dataSize+1:k*dataSize;
            yTest(ind) = net(data.testP(ind));
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', data.t(ind)', 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), data.t(ind));
        end
    else
        y = net.simulate(data.p);
        for k = 1:numTests
            ind = (k-1)*dataSize+1:k*dataSize;
            yTest(ind) = net.simulate(data.testP(ind));
            fitTest(k) = nnfw.goodnessOfFit(yTest(ind)', t(ind)', 'NRMSE');
            dbTest(k) = daviesBouldin(yTest(ind), t(ind));
        end
    end

    fit = nnfw.goodnessOfFit(y', data.t', 'NRMSE');
    db = daviesBouldin(y, data.t);

end

