clear;
close all;

%% set options
numNeurons = 2;
maxIter = 50;
useToolbox = true;
plotMeanOnly = false;
% select test part
idPtidC = 143;

%% prepare data

% load signal transformation data
tb_kt2 = sigTrans_loadData(idPtidC, 'kt2', 'y');
tb_kt3 = sigTrans_loadData(idPtidC, 'kt3', 'y');

% separate test data from train data
[values, indexes] = nnfw.Util.separateTrainingValues(tb_kt2, tb_kt2, 0.2, 0);
input = values{1,1};
testP = values{2,1};

% extract number of test values
numInputs = length(find(indexes{1,1}));
numTests = length(indexes{2,1});

% extract data size
dataSize = size(input,1);

% calculate mean of reference test bench
mean_tb_kt3 = mean(tb_kt3, 2);

[p, t] = prepareDataSISO(input', mean_tb_kt3');

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
simData.t = t;
simData.numTests = numTests;
simData.size = dataSize;

[y, yTest, fit, fitTest, db, dbTest] = simSISO(net, simData, useToolbox);

%% plot simulated data

plotData.figureNr = 2;
[plotData.title, plotData.xLabel, plotData.yLabel] = loadPlotData(idPtidC);
plotData.lgInput = 'ANN';
plotData.lgTestInput = 'Test Data';

plotData.lwInput = 1;
plotData.lwTestInput = 1;
plotData.colorInput = [0.0 1.0 0.0];
plotData.colorTestInput = [1.0 0.0 0.0];

plotData.y = y;
plotData.yTest = yTest;
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

plotOrigData.colorTB1 = 'm';
plotOrigData.colorTB2 = 'c';
plotOrigData.colorMeanTB1 = 'k';
plotOrigData.colorMeanTB2 = 'b';
plotOrigData.lineStyleMeanTB1 = '--';
plotOrigData.lineStyleMeanTB2 = '--';

plotCommon(plotOrigData);