% clear all;
clear;
clc;

% --------------------------------------
% load dataset
% --------------------------------------
% load house_dataset;
% load simplefit_dataset;
% load bodyfat_dataset;
% p = (-2:.1:2);
% % p = (-5:.1:5);
% t = cos(pi*p/2);
% 
% p = houseInputs;
% t = houseTargets;
% 
% p = simplefitInputs;
% t = simplefitTargets;
% 
% p = bodyfatInputs;
% t = bodyfatTargets;

% p = [-2:.1:2; -2:.1:2];
% t = [sin(pi*p(1,:)/2); cos(pi*p(2,:)/2)];

[p,t] = simpleclass_dataset;

% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
% % net1 = feedforwardnet([23 35]);
% net1 = feedforwardnet();
% % % net1.layers{1}.size = 14;
% % % net1.layers{1}.transferFcn = 'logsig';
% % net1.inputs{1}.processFcns = {}; % delete input preprocessing functions
% % net1.outputs{net1.numLayers}.processFcns = {}; % delete output postprocessing functions
% % net1.trainParam.max_fail = 10;
% % %net1.trainFcn = 'traingd';
% net1 = train(net1,p,t);
% % toolbox = net1(p);
% ba = net1(p);
% ba2 = zeros(size(ba));
% % extract indexes of max value of each column
% [~, i] = max(ba, [], 1);
% % map max values to given value
% ba2(sub2ind(size(ba), i, 1:length(i))) = 1;

% --------------------------------------
% init nn-framework
% --------------------------------------
% net = nnfw.FeedForward([20 10 5]);
% net = nnfw.FeedForward([20 10]);
% net = nnfw.FeedForward(120); % relativ guter fit
net = nnfw.FeedForward(10);
net.configure(p,t);
% net.outputs{net.numLayers}.f = nnfw.Util.Activation.TANH;
% net.layers{1}.f = nnfw.Util.Activation.LOGSIG;

% --------------------------------------
% prepare input/target
% --------------------------------------
% [in,net.minmaxInputSettings] = nnfw.Util.minmaxMapping(p);
% [tn,net.minmaxTargetSettings] = nnfw.Util.minmaxMapping(t);

% --------------------------------------
% train network
% --------------------------------------
% ba = zeros(1,length(in));
[E, ~, output, lambda, jacobian] = net.train(p,t);
% ba = nnfw.Util.minmaxMappingRevert(net.simulate(in), net.minmaxTargetSettings);
ba = net.simulate(p);
ba2 = zeros(size(ba));
% extract indexes of max value of each column
[~, i] = max(ba, [], 1);
% map max values to given value
ba2(sub2ind(size(ba), i, 1:length(i))) = 1;
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
% figure(2);
% hold on
% 
% plot(t(1,:), 'r'); % target
% plot(t(2,:), 'b'); % target
% plot(ba(1,:), 'y'); % ba
% plot(ba(2,:), 'g'); % ba
% legend('target sin','target cos', 'ba sin', 'ba cos');
% 
% plot(t, 'r'); % target
% % plot(toolbox, 'k'); % toolbox
% plot(ba, 'g'); % ba
% legend('target','toobox','ba');

% plot(p(1,:),p(2,:),'+');
% plot(ba(1,:),ba(2,:),'or');
% hold off

figure(3);
plotconfusion(t,ba2);