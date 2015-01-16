function [fit, meanFitTest, meanDbTest] = sigTrans_KT2_auf_KT3(net, useToolbox, idPtidC, testIndexes)
    %% prepare data
    % load signal transformation data
    sigTrans_loadData;

    % select test part
%     idPtidC = 137;

    % kt2
    dataindexKT2 = ismember(clusteringData(idPtidC).testbench, 'kt2');
    % kt3
    dataindexKT3 = ismember(clusteringData(idPtidC).testbench, 'kt3');

    yMeanKT3 = mean(clusteringData(idPtidC).yData(:, dataindexKT3), 2);

    p = clusteringData(idPtidC).yData(:,dataindexKT2)';
    t = yMeanKT3';

    % choose test data
    numTest = length(testIndexes);
    testDataInput = p(testIndexes, :);
    testDataSize = size(testDataInput,2);
    testDataInput = reshape(testDataInput',1,numTest*testDataSize);
    p = p(~ismember(1:size(p,1),testIndexes), :);

    inputRows = size(p,1);
    dimension = inputRows*size(p,2);
    p = reshape(p', 1, dimension);
    t = repmat(t, 1, 44);

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

    fit = nnfw.goodnessOfFit(y', t', 'NRMSE')

    %% extrapolation

    testDataSimulate = zeros(1, length(testDataInput));
    fitTest = zeros(1, numTest);
    dbTest = zeros(1, numTest);
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
        for k = 1:inputRows
            plot(clusteringData(idPtidC).xData(:,dataindexKT3)', y((k-1)*testDataSize+1:k*testDataSize), 'g', 'LineWidth', 2);
        end
        %plot mean data
        plot(clusteringData(idPtidC).xData(:,dataindexKT3)', yMeanKT3', '--r');
        %plot test data
        for k = 1:numTest-1
            plot(clusteringData(idPtidC).xData(:,dataindexKT3)', testDataSimulate((k-1)*testDataSize+1:k*testDataSize), 'r');
        end

        legend('ANN', 'KT3 Mean', 'Test Data');
    hold off;
    plot_kt2_to_kt3(idPtidC);
end