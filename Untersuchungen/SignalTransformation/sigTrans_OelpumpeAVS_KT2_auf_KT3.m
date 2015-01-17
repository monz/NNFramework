clear;
close all;

%% prepare data
% load signal transformation data
sigTrans_loadData;

% select test part
options.idPtidC = 143;

%% train network
% toolbox
net = feedforwardnet(2);
net.trainParam.epochs = 30;
options.useToolbox = true;

% nnfw
% net = nnfw.FeedForward(2);
% net.optim.maxIter = 40;
% options.useToolbox = false;

options.plotMeanOnly = false;
options.targetMean = true;
options.testIndexes = [43 44 45 46 49 50 51 52 53];

[~, ~, ~, net] = sigTrans_KT2_auf_KT3(net, options);

%% extrapolation
dataindexKT2 = ismember(clusteringData(options.idPtidC).testbench, 'kt2');
dataindexKT3 = ismember(clusteringData(options.idPtidC).testbench, 'kt3');
allKT2Data = clusteringData(options.idPtidC).yData(:, dataindexKT2);

extraData = allKT2Data(:,1);
extraTestData = allKT2Data(:,options.testIndexes(1));

% modify data
lengthExtra = length(extraData);
randIndexes = ceil(rand(1, 50)*lengthExtra);
extraData(randIndexes) = extraData(randIndexes)+rand*(max(max(extraData)*1.3));
% extraTestData(randIndexes) = extraTestData(randIndexes)+rand*(max(max(extraTestData)*1.3));
extraTestData(350:450) = extraTestData(350:450).*1.15;

figure(3)
hold on
    plot(clusteringData(options.idPtidC).xData(:,dataindexKT3)', extraData, 'y');
    plot(clusteringData(options.idPtidC).xData(:,dataindexKT3)', extraTestData, 'c');
hold off

% simulate
if options.useToolbox
    y = net(extraData');
    yTest = net(extraTestData');
else
    y = net.simulate(extraData');
    yTest = net.simulate(extraTestData');
end
% plot
figure(2);
hold on
    plot(clusteringData(options.idPtidC).xData(:,dataindexKT3)', y, 'y');
    plot(clusteringData(options.idPtidC).xData(:,dataindexKT3)', yTest, 'c');
hold off