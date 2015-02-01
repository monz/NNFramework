%%% -----------------------------------------------------------------------
% =========================================================================
%   Function Approximator
%   Train Method: MIMO
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
close all;

%% set options

numNeurons = 5;
maxIter = 50;
useToolbox = false;
figureNr = 2;

% --------------------------------------
% init training values
% --------------------------------------
p = [-2:.1:2; -2:.1:2];
t = [sin(pi*p(1,:)/2); cos(pi*p(2,:)/2)];


%% init and train neural network
if useToolbox
    net = feedforwardnet(numNeurons);
    net.trainParam.epochs = maxIter;
    net = train(net,p,t);
    y_d = net(p);
else
    net = nnfw.FeedForward(numNeurons);
    net.configure(p,t);
    net.optim.maxIter = maxIter;
    [E, ~, output, jacobian] = net.train(p,t);
    y_d = net.simulate(p); 
end

%% plot

figure(figureNr);
hold on
    plot(t(1,:), 'r'); % Target 1
    plot(t(2,:), 'b'); % Target 2
    plot(y_d(1,:), 'y'); % ANN 1
    plot(y_d(2,:), 'g'); % ANN 2
    legend('target sin','target cos', 'ANN sin', 'ANN cos');
hold off