clear;
close all;

%% prepare data
% load signal transformation data
sigTrans_loadData;

% select test part
idPtidC = 143;

%% train network
% toolbox
% net = feedforwardnet(1);
% net.trainParam.epochs = 30;

% nnfw
net = nnfw.FeedForward(2);
net.optim.maxIter = 40;

useToolbox = false;
testDataInd = [43 44 45 46 49 50 51 52 53];

sigTrans_KT2_auf_KT3(net, useToolbox, idPtidC, testDataInd);
