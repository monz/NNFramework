function [ outData ] = simSISO_NARX_Time( net, data )
%SIMSISO Summary of this function goes here
%   Detailed explanation goes here

    numTests = data.numTests;
    shift = data.shift;
    dataSize = data.size;
    inputRows = data.inputRows;
    targetSize = length(data.t);
    yTest = zeros(inputRows, numTests*(dataSize-shift));
    fitTest = zeros(inputRows, numTests);
    dbTest = zeros(1, numTests);
    delayNet = data.delayNet;
    
    % simulate normal input data
    yp = net(data.pr, data.Pi);
    
    % prepare extrapolation data for simulation
    y = con2seq(data.t(:, 1:dataSize));
    u = con2seq(data.extraP);
    if delayNet
        [pr,Pi, ~, ~, ~] = preparets(net,u,y);
    else
        [pr,Pi, ~, ~, ~] = preparets(net,u,{},y);
    end
    % simulate extraplotion data
    ypExtra = net(pr, Pi);

    % simulate test input data
    indTest = 0;
    for k = 1:numTests
        indTest = indTest(end)+1:indTest(end)+dataSize-shift;
        ind = (k-1)*dataSize+1:k*dataSize;
        indEndTarget = min(targetSize, k*dataSize);
        indStartTarget = (indEndTarget - (dataSize - shift)) + 1;
        indStartPrepareTarget = (indEndTarget - dataSize) +1;
        
        % prepare test data for simulation
        y = con2seq(data.t(:, indStartPrepareTarget:indEndTarget));
        u = con2seq(data.testP(:, ind));
        if delayNet
            [pr,Pi, ~, ~, ~] = preparets(net,u,y);
        else
            [pr,Pi, ~, ~, ~] = preparets(net,u,{},y);
        end
        % simulate
        ypTest = net(pr, Pi);
        % convert result
        yTest(:, indTest) = cell2mat(ypTest);
        % rate result
        fitTest(:, k) = nnfw.goodnessOfFit(yTest(:, indTest)', data.t(:, indStartTarget:indEndTarget)', 'NRMSE');
        dbTest(k) = daviesBouldin(yTest(:, indTest), data.t(:, indStartTarget:indEndTarget));
    end

    % convert results
    y = cell2mat(yp);
    yExtra = cell2mat(ypExtra);
    
    outData.fit = nnfw.goodnessOfFit(y', data.t(:, shift+1:end)', 'NRMSE');
    outData.fitExtra = nnfw.goodnessOfFit(yExtra', data.t(:, shift+1:dataSize)', 'NRMSE');
    outData.db = daviesBouldin(y, data.t(:, shift+1:end));
    outData.dbExtra = daviesBouldin(yExtra, data.t(:, shift+1:dataSize));
    
    outData.y = y;
    outData.yTest = yTest;
    outData.yExtra = yExtra;
    outData.fitTest = fitTest;
    outData.dbTest = dbTest;

end