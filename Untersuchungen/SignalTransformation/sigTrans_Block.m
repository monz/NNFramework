clear;
close all;

%% set options

% network options
numNeurons = [2];
maxIter = 15;
useToolbox = true;

% data options
trainInputMean = false;
trainTargetMean = false;
maxDimension = 400;

% plot options
plotMeanOnly = false;
plotReferenceOnly = true;
figureNr = 2;
plotName = 'Block';

% select test part
idPtidC = 143;
tb1 = 'kt4';
tb2 = 'kt3';

%% load path
addpath(genpath(pwd));

%% prepare data

% load signal transformation data
tb1Data = sigTrans_loadData(idPtidC, tb1, 'y');
tb2Data = sigTrans_loadData(idPtidC, tb2, 'y');

% separate test data from train data
[values, indexes] = nnfw.Util.separateTrainingValues(tb1Data, tb1Data, 0.99, 0);
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


% extract data size
dataSize = size(input,1);

% prepare extrapolated data
extraData = testData(:,1)*1.2; % 20-percent above normal input value
% extraData = rand(dataSize,1)*50; % 20-percent above normal input value

% extract number of test values
numTests = length(indexes{2,1});
% numTests = 1;

[p, t, testP, extraP] = prepareDataBlock(input', target', testData', extraData');

%% train network
numNets = ceil(size(p,1) / maxDimension);
nets = cell(numNets,1);

missingValues = max(dataSize, dataSize-maxDimension);
for k = 1:numNets
    startInd = (k-1) * maxDimension + 1;
    endInd = min((startInd - 1) + missingValues, k*maxDimension);
    missingValues = missingValues - min(maxDimension, missingValues);
    if useToolbox
        net = feedforwardnet(numNeurons);
        net.trainParam.epochs = maxIter;
        net.inputs{1}.processFcns = {};
        net.outputs{2}.processFcns = {};
        nets{k} = train(net,p(startInd:endInd),t(startInd:endInd));
    else
        net = nnfw.FeedForward(numNeurons);
        net.optim.maxIter = maxIter;
        net.configure(p(startInd:endInd),t(startInd:endInd));
        net.train(p(startInd:endInd),t(startInd:endInd));
        nets{k} = net;
        close all;
    end
end

%% simulate 

simData.p = p;
simData.testP = testP;
simData.extraP = extraP;
simData.t = t;
simData.numTests = numTests;
simData.size = dataSize;
simData.numNets = numNets;
simData.maxDimension = maxDimension;

net = nets{k};
[simOutData] = simBlock(nets, simData, useToolbox);
    
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

plotData.figureNr = figureNr;
[plotData.title, plotData.xLabel, plotData.yLabel] = loadPlotData(idPtidC);
plotData.lgInput = 'ANN';
plotData.lgTestInput = 'Test Data';
plotData.lgExtraInput = 'Extra Data';

plotData.lwInput = 1;
plotData.lwTestInput = 2;
plotData.lwExtraInput = 2;
plotData.colorInput = [0.0 1.0 0.0];
plotData.colorTestInput = [1.0 0.0 0.0];
plotData.colorExtraInput = [0.5 0.5 0.5];

plotData.y = y;
plotData.yTest = yTest;
plotData.yExtra = yExtra;
plotData.xAxis = sigTrans_loadData(idPtidC, tb2, 'x');
plotData.size = dataSize;
plotData.numInputs = numInputs;
plotData.numTest = numTests;

plotSISO(plotData);

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

%% save figures
outDir = 'figures';
if ~exist(outDir, 'dir')
  mkdir(outDir);
end

ext = {'fig','png'};
% save figure with all data
close all;

plotOrigData.meanOnly = false;
plotOrigData.referenceOnly = false;

plotSISO(plotData);
plotCommon(plotOrigData);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

for k = 1:length(ext)
    saveas(gcf, sprintf('%s/%d_%s_all_%s.%s', outDir, idPtidC, plotName, datestr(now,'dd.mm.yyyy_HHMM'), ext{k}));
end

% save figure with reference data only
close all;

plotOrigData.meanOnly = false;
plotOrigData.referenceOnly = true;

plotSISO(plotData);
plotCommon(plotOrigData);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

for k = 1:length(ext)
    saveas(gcf, sprintf('%s/%d_%s_reference_%s.%s', outDir, idPtidC, plotName, datestr(now,'dd.mm.yyyy_HHMM'), ext{k}));
end

% save figure with mean data only
close all;

plotOrigData.meanOnly = true;
plotOrigData.referenceOnly = false;

plotSISO(plotData);
plotCommon(plotOrigData);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

for k = 1:length(ext)
    saveas(gcf, sprintf('%s/%d_%s_mean_%s.%s', outDir, idPtidC, plotName, datestr(now,'dd.mm.yyyy_HHMM'), ext{k}));
end

close all;

%% add original extrapolation data to plot
% figure(plotData.figureNr);
% hold on;
%     plot(plotData.xAxis, extraData, ':k', 'LineWidth',2);
% hold off

%% re-plot with mean values only
% plotOrigData.meanOnly = 1;
% plotSISO(plotData);
% plotCommon(plotOrigData);
