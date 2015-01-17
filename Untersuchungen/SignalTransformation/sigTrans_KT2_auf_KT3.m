function [fit, meanFitTest, meanDbTest, net] = sigTrans_KT2_auf_KT3(net, options)
    %% prepare data
    % load signal transformation data
    sigTrans_loadData;

    useToolbox = options.useToolbox;
    plotMeanOnly = options.plotMeanOnly;
    targetMean = options.targetMean;
    blockTrain = options.blockTrain;
    idPtidC = options.idPtidC;
    testIndexes = options.testIndexes;
    numTest = length(testIndexes);

    % kt2
    dataindexKT2 = ismember(clusteringData(idPtidC).testbench, 'kt2');
    % kt3
    dataindexKT3 = ismember(clusteringData(idPtidC).testbench, 'kt3');

    p = clusteringData(idPtidC).yData(:,dataindexKT2)';
    
    numInputs = size(p,1)-numTest;
    inputDataLength = size(p,2);
    dimension = numInputs*inputDataLength;
    if targetMean
        yMeanKT3 = mean(clusteringData(idPtidC).yData(:, dataindexKT3), 2);
        t = yMeanKT3';
    else
        t = clusteringData(idPtidC).yData(:,dataindexKT3)';
        numTargets = size(t,1);
        if numTargets < numInputs
            tTmp = zeros(numInputs,length(t));
            tTmp(1:numTargets,:) = t(1:numTargets,:);
            tTmp(numTargets+1:numInputs,:) = t(1:numInputs-numTargets,:);
            t = tTmp;
        elseif numTargets > numInputs
            t = t(1:numInputs,:);
        end
    end

    % choose test data
    testDataInput = p(testIndexes, :);
    testDataSize = size(testDataInput,2);
    p = p(~ismember(1:size(p,1),testIndexes), :);

    % prepare block training if necessary
    if blockTrain
        testDataInput = testDataInput';
        p = p';
        t = t';
        if targetMean
            t = repmat(t, 1, numInputs);
%             t(:, 1:2:numInputs) = t(:, 1:2:numInputs)*1.03;
            t(:, 1:3:numInputs) = t(:, 1:3:numInputs)*1.01;
        end
    else
        % no block train, prepare data for SISO
        testDataInput = reshape(testDataInput',1,numTest*testDataSize);
        p = reshape(p', 1, dimension);
        if targetMean
            t = repmat(t, 1, numInputs);
        else
            t = reshape(t', 1, dimension);
        end
    end

    %% train network
    
    if useToolbox
        % toolbox
        net = train(net,p,t);
        y = net(p);
    else
        % nnfw
        net.configure(p,t);
        net.train(p,t);
        y = net.simulate(p);
    end

    if blockTrain
        fit = nnfw.goodnessOfFit(y, t, 'NRMSE')
    else
        fit = nnfw.goodnessOfFit(y', t', 'NRMSE')
    end

    %% simulate test data

    testDataSimulate = zeros(1, length(testDataInput));
    fitTest = zeros(1, numTest);
    dbTest = zeros(1, numTest);
    
    if blockTrain
        testDataSimulate = zeros(size(testDataInput));
        for k = 1:numTest
            if useToolbox
                testDataSimulate(:,k) = net(testDataInput(:,k));
            else
                testDataSimulate(:,k) = net.simulate(testDataInput(:,k));
            end
            fitTest(k) = nnfw.goodnessOfFit(testDataSimulate(:,k), t(:,k), 'NRMSE');
            dbTest(k) = daviesBouldin(testDataSimulate(:,k), t(:,k));
        end
    else
        for k = 1:numTest
            ind = (k-1)*testDataSize+1:k*testDataSize;
            if useToolbox
                testDataSimulate(ind) = net(testDataInput(ind));
            else
                testDataSimulate(ind) = net.simulate(testDataInput(ind));
            end
            fitTest(k) = nnfw.goodnessOfFit(testDataSimulate(ind)', t(ind)', 'NRMSE');
            dbTest(k) = daviesBouldin(testDataSimulate(ind), t(ind));
        end
    end

    % mean fitTest; dbTest
    meanFitTest = mean(fitTest)
    meanDbTest = mean(dbTest)

    %% plot
    figure(2);
    hold on
        title('Oelpumpe KT2 auf KT3');
        xlabel(clusteringData(idPtidC).xLabel);
        ylabel(clusteringData(idPtidC).yLabel);
        % plot(p',y','r'); % Verhältnis Eingang, Simulation
        for k = 1:numInputs
            plot(clusteringData(idPtidC).xData(:,dataindexKT3)', y((k-1)*testDataSize+1:k*testDataSize), 'Color', [0.0 1.0 0.5], 'LineWidth', 2);
        end
        %plot mean data
        if options.targetMean
            plot(clusteringData(idPtidC).xData(:,dataindexKT3)', yMeanKT3', '--r');
        end
        %plot test data
        for k = 1:numTest-1
            plot(clusteringData(idPtidC).xData(:,dataindexKT3)', testDataSimulate((k-1)*testDataSize+1:k*testDataSize), 'r');
        end

        legend('ANN', 'KT3 Mean', 'Test Data');
    hold off;
    plot_kt2_to_kt3(idPtidC, plotMeanOnly);
end