%% signal transformation NARX

clear;
close all;

%% set options
numNeurons = [10 10];
maxIter = 100;
delay1 = [1:2];
delay2 = [1:2];
plotMeanOnly = false;
plotReferenceOnly = true;
trainInputMean = false;
trainTargetMean = false;
addTimeInput = true;
% select test part
idPtidC = 81;
tb1 = 'kt2';
tb2 = 'kt3';

%% prepare data

% load signal transformation data
tb1Data = sigTrans_loadData(idPtidC, tb1, 'y');
tb2Data = sigTrans_loadData(idPtidC, tb2, 'y');

% load times data
times = sigTrans_loadData(idPtidC, tb2, 'x');
% all times are the same, therefore reduce to a single time slice
times = times(:,1);

% separate test data from train data
[values, indexes] = nnfw.Util.separateTrainingValues(tb1Data, tb1Data, 0.2, 0);
input = values{1,1};
testData = values{2,1};

% prepare input data
if trainInputMean
    % calculate mean of input test bench
    input = mean(input, 2);
    numInputs = 1;
else  
    numInputs = length(find(indexes{1,1}));
end

% prepare target data
if trainTargetMean
    % calculate mean of reference test bench
    mean_tb2 = mean(tb2Data, 2);
    target = mean_tb2;
else
    target = tb2Data;
end

% prepare extrapolated data
extraData = testData(:,1)*1.2; % 20-percent above normal input value

% extract number of test values
numTests = length(indexes{2,1});

% extract data size
dataSize = size(input,1);

if addTimeInput
    [p, t, testP, extraP] = prepareDataSISO_Time(input', target', testData', extraData', times');
else
    [p, t, testP, extraP] = prepareDataSISO(input', target', testData', extraData');
end

% extract input rows
inputRows = size(p,1);

%% train network

y = con2seq(t);
u = con2seq(p);

net = narxnet(delay1,delay2,numNeurons);
net.trainParam.epochs = maxIter;

[pr,Pi,Ai,tr, ~, shift] = preparets(net,u,{},y);

net = train(net,pr,tr,Pi);

%% simulate

simData.pr = pr;
simData.Pi = Pi;
simData.testP = testP;
simData.extraP = extraP;
simData.t = t;
simData.numTests = numTests;
simData.inputRows = inputRows;
simData.size = dataSize;
simData.shift = shift;

[simOutData] = simSISO_NARX_Time(net, simData);

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
fitTestMean = mean(fitTest, 2);
dbTestMean = mean(dbTest);

%% plot simulated data

plotData.figureNr = 4;
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
plotData.shift = shift;
plotData.xAxis = sigTrans_loadData(idPtidC, tb2, 'x');
plotData.size = dataSize;
plotData.numInputs = numInputs;
plotData.numTest = numTests;

plotSISO_NARX_Time(plotData);

%% plot original data
plotOrigData.figureNr = plotData.figureNr;
plotOrigData.meanOnly = plotMeanOnly;
plotOrigData.referenceOnly = plotReferenceOnly;

plotOrigData.xAxisTB1 = sigTrans_loadData(idPtidC, tb1, 'x');
plotOrigData.xAxisTB2 = sigTrans_loadData(idPtidC, tb2, 'x');
plotOrigData.yAxisTB1 = sigTrans_loadData(idPtidC, tb1, 'y');
plotOrigData.yAxisTB2 = sigTrans_loadData(idPtidC, tb2, 'y');
plotOrigData.yAxisMeanTB1 = mean(plotOrigData.yAxisTB1,2);
plotOrigData.yAxisMeanTB2 = mean(plotOrigData.yAxisTB2,2);
        
plotOrigData.lgTB1 = tb1;
plotOrigData.lgTB2 = tb2;
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
% plotSISO_NARX_Time(plotData);
% plotCommon(plotOrigData);
