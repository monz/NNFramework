% clear all;
clear;
clc;

% --------------------------------------
% load dataset
% --------------------------------------
% load house_dataset;
% load simplefit_dataset;
load bodyfat_dataset;
% p = (-2:.1:2);
p = (-5:.1:5);
t = cos(pi*p/2);
% 
% p = houseInputs;
% t = houseTargets;
% 
% p = simplefitInputs;
% t = simplefitTargets;

% p = bodyfatInputs;
% t = bodyfatTargets;

% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
% net1 = feedforwardnet([23 35]);
% net1 = feedforwardnet();
% % net1.layers{1}.size = 14;
% % net1.layers{1}.transferFcn = 'logsig';
% net1.inputs{1}.processFcns = {}; % delete input preprocessing functions
% net1.outputs{net1.numLayers}.processFcns = {}; % delete output postprocessing functions
% net1.trainParam.max_fail = 10;
% %net1.trainFcn = 'traingd';
% net1 = train(net1,p,t);
% toolbox = net1(p);

% --------------------------------------
% init nn-framework
% --------------------------------------
net = nnfw.FeedForward(1, 2, 1);
net.configure(p,t);
% net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
net.layers{1}.size = 30;
% net.layers{2}.size = 10; % set layer 1 numOfNeurons to 10
weights = rand(net.getNumWeights(),1);
net.setWeights(weights);

% --------------------------------------
% prepare input/target
% --------------------------------------
[in,net.minmaxInputSettings] = nnfw.Util.minmaxMapping(p);
[tn,net.minmaxTargetSettings] = nnfw.Util.minmaxMapping(t);

% --------------------------------------
% train network
% --------------------------------------
% ba = zeros(1,length(in));
% E = 1;
% EMin = 1e16;
% count = 100;
% weightsRand = rand(net.getNumWeights(),count);
% weightsMin = zeros(net.getNumWeights(),1);
% mu = 0.3;
% while E > 0 && count > 0
    [E, ~, output, lambda, jacobian] = net.train(in,tn);
%     E
%     if E < EMin
%        EMin = E;
%        weightsMin = net.getWeightVector();
%     end
%     weights = weightsMin;
%     weights = weights - mu*rand(net.getNumWeights(),1);
% %     weights = weightsRand(:,count);
%     net.setWeights(weights);
%     count = count-1
% end
% net.setWeights(weightsMin);
% EMin
ba = nnfw.Util.minmaxMappingRevert(net.simulate(in), net.minmaxTargetSettings);
% ba = net.simulate(in);

% --------------------------------------
% goodness of fit
% --------------------------------------
% nnfw.goodnessOfFit(toolbox, t, 'mse')
% nnfw.Util.CostFunction.MSE.f(toolbox, t)

% nnfw.goodnessOfFit(ba, t, 'mse')
% nnfw.Util.CostFunction.MSE.f(ba, t)

% nnfw.goodnessOfFit(ba, toolbox, 'mse')
% nnfw.Util.CostFunction.MSE.f(ba, toolbox)

% --------------------------------------
% plot
% --------------------------------------
figure(2);
hold on
plot(t, 'r'); % target
% plot(toolbox, 'k'); % toolbox
plot(ba, 'g'); % ba
legend('target','toobox','ba');
hold off