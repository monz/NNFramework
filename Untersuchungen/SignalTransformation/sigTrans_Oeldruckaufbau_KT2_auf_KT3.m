%% Oeldruckaufbau

clear;
close all;

%% prepare data
% load signal transformation data
sigTrans_loadData;

% select test part
options.idPtidC = 83;

%% train network
% toolbox
net = feedforwardnet(2);
net.trainParam.epochs = 30;
options.useToolbox = true;

% nnfw
% net = nnfw.FeedForward(2);
% net.optim.maxIter = 30;
% options.useToolbox = false;

options.plotMeanOnly = true;
options.targetMean = false;
options.blockTrain = true;
options.testIndexes = [43 44 45 46 49 50 51 52 53];

[~, ~, ~, net] = sigTrans_KT2_auf_KT3(net, options);

% plot(net(rand(400,1)));