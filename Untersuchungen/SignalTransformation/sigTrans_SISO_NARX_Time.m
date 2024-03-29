%%% -----------------------------------------------------------------------
% =========================================================================
%   Signal Transformation Script
%   Train Method: SISO with Delay-Net/NARX-Net
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
%   delay1:             sets the input delay (for Delay- and NARX-Net)
%                       e.g. 1:2 or 1:10
%
%   delay2:             sets the target delay (for NARX-Net only)
%
%   delayNet:           if true use delayNet for training, if false switch
%                       to NARX-Net
%
%   narxVariant:        switches the training method of the NARX-Net
%                       can be 'open' or 'closed'
%
%   trainInputMean:     if true the mean of all input signals is calculated and
%                       only this mean value is used as input value for
%                       network training, if fasle all input signals are used
%                       for network training
%
%   trainTargetMean:    if true the mean of all target signals is calculated and
%                       only this mean value is used as target value for
%                       network training, if false all target signals are used
%                       for network training 
%
%   addTimeInput:       if true values of the x-axis gets added to the input.
%                       The input dimension is then two.
%
%   flipTime:           if true the input and target values get flipped from
%                       left to right, using the fliplr MATLAB method.
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
%
%   =======================================================================
%   the file name is composed as follows:
%   -----------------------------------------------------------------------
%   idPtidC_netType_<settings>_plotType_currDate.ext
%
%   settings:
%   numNeurons_tb1_tb2_trainInputMean_trainTargetMean_<options>
%
%   options:
%   flipTime_maxDimension_delay1
%   -----------------------------------------------------------------------
%   boolean values are mapped to 0,1 in the file name
%   =======================================================================
%%% -----------------------------------------------------------------------

clear;
close all;

%% set options

% network options
numNeurons = [10];
maxIter = 100;
delay1 = [1:20];
delay2 = [1:2];
delayNet = true;
narxVariant = 'closed';

% data options
trainInputMean = false;
trainTargetMean = false;
addTimeInput = false;
flipTime = true;

% plot options
plotMeanOnly = false;
plotReferenceOnly = false;
figureNr = 2;

% save figures options
saveFigures = false;
extensions = {'fig','png'};
outDir = 'figures';
netType = 'TimeDelay-NARX';

% select test part
idPtidC = 35;
tb1 = 'kt4';
tb2 = 'kt3';

%% load path
addpath(genpath(pwd));

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

% flip left right
if flipTime
    p = fliplr(p);
    t = fliplr(t);
    testP = fliplr(testP);
    extraP = fliplr(extraP);
end

% extract input rows
inputRows = size(p,1);

%% train network

y = con2seq(t);
u = con2seq(p);

if delayNet
    net = timedelaynet(delay1,numNeurons);
    net.trainParam.epochs = maxIter;
    [pr,Pi,Ai,tr,~,shift] = preparets(net,u,y);
    net = train(net,pr,tr,Pi,Ai);
else
    net = narxnet(delay1,delay2,numNeurons, narxVariant);
    net.trainParam.epochs = maxIter;

    [pr,Pi,Ai,tr, ~, shift] = preparets(net,u,{},y);

    net = train(net,pr,tr,Pi);
end

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
simData.delayNet = delayNet;

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

% reverse flip left right
if flipTime
    y = fliplr(y);
    yTest = fliplr(yTest);
    yExtra = fliplr(yExtra);
end

%% rate data
fitTestMean = mean(fitTest, 2);
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
plotData.shift = shift;
plotData.xAxis = sigTrans_loadData(idPtidC, tb2, 'x');
plotData.size = dataSize;
plotData.numInputs = numInputs;
plotData.numTest = numTests;
plotData.flipTime = flipTime;

plotSISO_NARX_Time(plotData);

%% plot original data
plotOrigData.figureNr = plotData.figureNr;
plotOrigData.meanOnly = plotMeanOnly;
plotOrigData.referenceOnly = plotReferenceOnly;

plotOrigData.shift = shift;
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
    data.plotSpecific = @plotSISO_NARX_Time;
    data.plotCommon = @plotCommon;

    data.timeFlip = flipTime;
    data.maxDimension = 0;
    data.delay1 = delay1;

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