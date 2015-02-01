%%% -----------------------------------------------------------------------
% =========================================================================
%   Manually Check Calculation of Jacobian Matrix
%   This script was only used for debugging purposes!
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
%   useToolbox:         if true uses MATLAB-NNToolbox for training, if false it
%                       uses the NN-Framework
%
%   figureNr:           sets the figure handle
%%% -----------------------------------------------------------------------

clear;
clc;

%% set options

numNeurons = 2;
useToolbox = true;
figureNr = 2;

% --------------------------------------
% init training values
% --------------------------------------
% p = [-2:.1:2; -2:.1:2];
p = [0.9; 0.9];
t = [sin(pi*p(1,:)/2); cos(pi*p(2,:)/2)];

%% init and train neural network
if useToolbox
    net = feedforwardnet(numNeurons);
    net = train(net,p,t);
    y_d = net(p);
else
    net = nnfw.FeedForward(numNeurons);
    net.configure(p,t);
    net.setWeights([0.75 0.69 -0.54 0.17 -0.23 0.47 0.18 0.71 0.38 0.72 0.11 0.33]');
    net.simulate(p, false);
    costFcn = net.makeCostFcn2(@nnfw.Util.componentError, p, t);
    [F, J] = costFcn(net.getWeightVector());
    [E, ~, output, jacobian] = net.train(p,t);
    y_d = net.simulate(p);
end

%% plot

% figure(figureNr);
% hold on
%     plot(t(1,:), 'r'); % Target 1
%     plot(t(2,:), 'b'); % Target 2
%     plot(y_d(1,:), 'y'); % ANN 1
%     plot(y_d(2,:), 'g'); % ANN 2
%     legend('Target sin','Target cos', 'ANN sin', 'ANN cos');
% hold off