%%% -----------------------------------------------------------------------
% =========================================================================
%   Signal Transformation Script
%   Train Method: SISO
% =========================================================================
%
% #########################################################################
%   DO NOT CHANGE CODE BELOW THE "set options" SECTION
% #########################################################################
%
%   Possible Settings:
%   ------------------
%
%   numNeurons:         sets the number of neurons and layers in the neural network
%                       e.g. single hidden layer with 10 neurons; 10
%                       or two hidden layer with each 10 neurons; [10 10]
%
%   maxIter:            sets the max number of training iterations
%                       - positive number - e.g. 50
%
%   useToolbox:         if true uses MATLAB-NNToolbox for training, if false it
%                       uses the NN-Framework
%
%   trainMeanInput:     if true the mean of all input signals is calculated and
%                       only this mean value is used as input value for
%                       network training, if fasle all input signals are used
%                       for network training
%
%   trainMeanTarget:    if true the mean of all target signals is calculated and
%                       only this mean value is used as target value for
%                       network training, if false all target signals are used
%                       for network training 
%
%   plotMeanOnly:       if true only the calculated mean value of input/target
%                       signals are plotted - this setting overrides the
%                       plotReferenceOnly option, which will be ignored then. If
%                       false all input/target signals are plotted.
%
%   plotReferenceOnly:  if true only the target signals are plotted, if
%                       false all input/target signals are plotted
%
%   figureNr:           sets the figure handle
%
%   Simulated values of the neural network gets always plotted!
%
%   saveFigures:        if true the plot options are ignored. Instead three
%                       plots with differen settings will be saved to disk.
%                       One with all data, reference data and mean only.
%                       The plots get saved as .png and .fig files.
%                       Additionally to the plot the workspace will be
%                       saved.
%
%   extensions:         file extensions of which type the figures are saved
%
%   outDir:             directory in which the files are saved
%
%   netType:            this is integrated in the file name to distinguish the
%                       different training methods if the plots get
%                       automatically saved to disk
%
%   idPtidC:            id to select the test bench from data set
%
%   tb1:                test bench one - can be one of {'kt2','kt3','kt4'}
%
%   tb2:                test bench two - possible values see tb1
%%% -----------------------------------------------------------------------

clear;
close all;

%% set options

% network options
numNeurons = [10];
maxIter = 100;
useToolbox = true;

% data options
trainInputMean = false;
trainTargetMean = true;

% plot options
plotMeanOnly = true;
plotReferenceOnly = true;
figureNr = 2;

% save figures options
saveFigures = false;
extensions = {'fig','png'};
outDir = 'figures';
netType = 'SISO';

% select test part
idPtidC = 137;
tb1 = 'kt2';
tb2 = 'kt3';

%% load path
addpath(genpath(pwd));

%% prepare data

% load signal transformation data
tb1Data = sigTrans_loadData(idPtidC, tb1, 'y');
tb2Data = sigTrans_loadData(idPtidC, tb2, 'y');

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

plotData.figureNr = figureNr;
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
if saveFigures
    data.plotSpecific = @plotSISO;
    data.plotCommon = @plotCommon;

    data.timeFlip = 0;
    data.maxDimension = 0;
    data.delay1 = 0;

    data.ext = extensions;
    data.outDir = outDir;
    data.date = datestr(now,'dd.mm.yyyy_HHMM');
    data.idPtidC = idPtidC;
    data.netType = netType;

    data.tb1 = tb1;
    data.tb2 = tb2;
    data.numNeurons = numNeurons;
    data.meanInput = trainInputMean;
    data.meanTarget = trainTargetMean;

    % actually save plots
    plotOrigData.meanOnly = false;
    plotOrigData.referenceOnly = false;
    savePlot(data, plotData, plotOrigData, 'all');

    plotOrigData.meanOnly = false;
    plotOrigData.referenceOnly = true;
    savePlot(data, plotData, plotOrigData, 'reference');

    plotOrigData.meanOnly = true;
    plotOrigData.referenceOnly = false;
    savePlot(data, plotData, plotOrigData, 'mean');

    %% save data to .mat file
    save(sprintf('%s/%d_%s_%s_%s_%s_%d_%d%s_%s.mat', ...
        data.outDir, data.idPtidC, data.netType, ...
        data.tb1, data.tb2, getNetSizeString(numNeurons), data.meanInput, data.meanTarget, ...
        getOptionsString(data), data.date));
end