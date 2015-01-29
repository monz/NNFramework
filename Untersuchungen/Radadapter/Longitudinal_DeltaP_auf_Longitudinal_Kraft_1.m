%%% -----------------------------------------------------------------------
% =========================================================================
%   Wheel behavior Study Script
%   Train Method: SISO with Feedforward
%
%   Longitudinal_DeltaP to Longitudinal_Kraft
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
%   plotValidateData:   if true the validation data gets simulated with the
%                       trained network and plotted to see how good the
%                       neural network is for generalization
%
%   figureNr:           sets the figure handle
%
%   trainFile:          name of file containing the training data
%
%   validateFile:       name of file containing the validation data
%%% -----------------------------------------------------------------------

clear;
close all;

%% set options

% neural net settings
numNeurons = 5;
maxIter = 50;
useToolbox = true;

% plot settings
figureNr = 2;
plotValidateData = true;

% load data settings

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';
validateFile = '03_APRBS_APK1_sim3_DRV_100Pct_RSP.tim';

traindata = loadDataF(trainFile);
validatedata = loadDataF(validateFile);

%% define input and target data

x = [traindata.Longitudinal_DeltaP'];
t = [traindata.Longitudinal_Kraft'];
xV = [validatedata.Longitudinal_DeltaP'];
tV = [validatedata.Longitudinal_Kraft'];

%% define and train neural network

if useToolbox
    net = feedforwardnet(numNeurons);
    net.trainParam.epochs = maxIter;
    net = train(net,x,t);
else
    net = nnfw.FeedForward(numNeurons);
    net.configure(x,t);
    net.optim.maxIter = maxIter;
    net.train(x,t);
end

%% simulate data
if useToolbox
    y = net(x);
else
    y = net.simulate(x);
end

if plotValidateData
    if useToolbox
        yVal = net(xV);
    else
        yVal = net.simulate(xV);
    end
end

%% plot results
figure(figureNr)
hold on
    title('Longidutinal DeltaP auf Longitudinal Kraft');
    plot(t,'r');
    plot(y,'g');
    if plotValidateData
        plot(tV,'c');
        plot(yVal,'k');
        legend('Target','ANN','Target Validation', 'ANN Validation');
    else
        legend('Simulation','ANN');
    end
hold off

%% rate results
fit = nnfw.goodnessOfFit(y',t','NRMSE')
if plotValidateData
    fitVal = nnfw.goodnessOfFit(yVal',tV','NRMSE')
end