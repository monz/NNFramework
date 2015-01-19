clear;
close all;

%% set options
numNeurons = 2;
maxIter = 50;
useToolbox = true;
plotMeanOnly = false;
trainTargetMean = false;
% select test part
idPtidC = 143;

%% prepare data

% load signal transformation data
tb_kt2 = sigTrans_loadData(idPtidC, 'kt2', 'y');
tb_kt3 = sigTrans_loadData(idPtidC, 'kt3', 'y');

% separate test data from train data
[values, indexes] = nnfw.Util.separateTrainingValues(tb_kt2, tb_kt2, 0.2, 0);
input = values{1,1};
testData = values{2,1};

% prepare extrapolated data
extraData = testData(:,1)*1.2; % 30-percent above normal input value

% extract number of test values
numInputs = length(find(indexes{1,1}));
numTests = length(indexes{2,1});

% extract data size
dataSize = size(input,1);

if trainTargetMean
    % calculate mean of reference test bench
    mean_tb_kt3 = mean(tb_kt3, 2);
    target = mean_tb_kt3;
else
    target = tb_kt3;
end

[p, t, testP, extraP] = prepareDataSISO(input', target', testData', extraData');

%% train network

if useToolbox
    net = feedforwardnet(numNeurons);
    net.trainParam.epochs = maxIter;
    net = train(net,p,t);
else
    net = nnfw.FeedForward(numNeurons);
    net.optim.maxIter = maxIter;
    net.configure(p,t);
    net.train(p,t);
end

%% simulate 
simData.p = p;
simData.testP = testP;
simData.extraP = extraP;
simData.t = t;
simData.numTests = numTests;
simData.size = dataSize;

[simOutData] = simSISO(net, simData, useToolbox);

y = simOutData.y;
yTest = simOutData.yTest;
yExtra = simOutData.yExtra;
fit = simOutData.fit;
fitTest = simOutData.fitTest;
fitExtra = simOutData.fitExtra;
db = simOutData.db;
dbTest = simOutData.dbTest;
dbExtra = simOutData.dbExtra;

%% rate data
fitTestMean = mean(fitTest);
dbTestMean = mean(dbTest);

%% plot simulated data

plotData.figureNr = 2;
[plotData.title, plotData.xLabel, plotData.yLabel] = loadPlotData(idPtidC);
plotData.lgInput = 'ANN';
plotData.lgTestInput = 'Test Data';
plotData.lgExtraInput = 'Extra Data';

plotData.lwInput = 1;
plotData.lwTestInput = 1;
plotData.lwExtraInput = 2;
plotData.colorInput = [0.0 1.0 0.0];
plotData.colorTestInput = [1.0 0.0 0.0];
plotData.colorExtraInput = [0.5 0.5 0.5];

plotData.y = y;
plotData.yTest = yTest;
plotData.yExtra = yExtra;
plotData.xAxis = sigTrans_loadData(idPtidC, 'kt3', 'x');
plotData.size = dataSize;
plotData.numInputs = numInputs;
plotData.numTest = numTests;

plotSISO(plotData);

%% plot original data
plotOrigData.figureNr = plotData.figureNr;
plotOrigData.meanOnly = plotMeanOnly;

plotOrigData.xAxisTB1 = sigTrans_loadData(idPtidC, 'kt2', 'x');
plotOrigData.xAxisTB2 = sigTrans_loadData(idPtidC, 'kt3', 'x');
plotOrigData.yAxisTB1 = sigTrans_loadData(idPtidC, 'kt2', 'y');
plotOrigData.yAxisTB2 = sigTrans_loadData(idPtidC, 'kt3', 'y');
plotOrigData.yAxisMeanTB1 = mean(plotOrigData.yAxisTB1,2);
plotOrigData.yAxisMeanTB2 = mean(plotOrigData.yAxisTB2,2);
        
plotOrigData.lgTB1 = 'KT2';
plotOrigData.lgTB2 = 'KT3';
plotOrigData.lgInput = 'ANN';
plotOrigData.lgTestInput = 'Test Data';
plotOrigData.lgExtraInput = 'Extra Data';

plotOrigData.colorTB1 = 'm';
plotOrigData.colorTB2 = 'c';
plotOrigData.colorMeanTB1 = 'k';
plotOrigData.colorMeanTB2 = 'b';
plotOrigData.lineStyleMeanTB1 = '--';
plotOrigData.lineStyleMeanTB2 = '--';

plotCommon(plotOrigData);

%% add original extrapolation data to plot
% figure(plotData.figureNr);
% hold on;
%     plot(plotData.xAxis, extraData, ':k', 'LineWidth',2);
% hold off

%% re-plot with mean values only
% plotOrigData.meanOnly = 1;
% plotSISO(plotData);
% plotCommon(plotOrigData);
