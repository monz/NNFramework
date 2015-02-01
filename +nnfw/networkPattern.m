%%% -----------------------------------------------------------------------
% =========================================================================
%   Pattern Matching
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
%   figureNr:           sets the figure handle
%%% -----------------------------------------------------------------------

clear;
clc;

%% set options

numNeurons = 5;
maxIter = 50;
useToolbox = false;
figureNr = 2;

% --------------------------------------
% init training values
% --------------------------------------
[p,t] = simpleclass_dataset;

%% init train neural network

if useToolbox
    net = feedforwardnet(numNeurons);
    net.trainParam.epochs = maxIter;
    net = train(net,p,t);
    y_d = net(p);
else
    isPatternNet = true;
    net = nnfw.FeedForward(numNeurons, isPatternNet);
    net.configure(p,t);
    net.optim.maxIter = maxIter;
    [E, ~, output, jacobian] = net.train(p,t);
    y_d = net.simulate(p); 
end

%% plot

figure(figureNr);
hold on
    plotconfusion(t,y_d);
hold off