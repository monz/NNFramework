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
% t = cos(pi*p/2);
% 
% p = houseInputs;
% t = houseTargets;
% 
% p = simplefitInputs;
% t = simplefitTargets;

p = bodyfatInputs;
t = bodyfatTargets;


% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
% net1 = feedforwardnet([23 35]);
net1 = feedforwardnet();
% net1.layers{1}.size = 14;
% net1.layers{1}.transferFcn = 'logsig';
net1.inputs{1}.processFcns = {}; % delete input preprocessing functions
net1.outputs{net1.numLayers}.processFcns = {}; % delete output postprocessing functions
net1.trainParam.max_fail = 10;
%net1.trainFcn = 'traingd';
net1 = train(net1,p,t);
toolbox = net1(p);

% --------------------------------------
% init nn-framework
% --------------------------------------
net = nnfw.FeedForward(1, 2, 1);
% net.layers{1}.f = @logsig;
% net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
net.layers{2}.size = 10; % set layer 2 numOfNeurons to 10
net.IW{1} = net1.IW{1};
net.LW{2,1} = net1.LW{2,1};
% net.LW{3,2} = net1.LW{3,2};
net.b{1,1} = net1.b{1,1};
net.b{2,1} = net1.b{2,1};
% net.b{3,1} = net1.b{3,1};
ba = zeros(1,length(p));
ind = 1;
net.train(p,t);
ba = net.simulate(p);
% for k = 1:length(p)
% for k = p;
% %     out = net.train(p(:,k));
% %     ba(ind) = out;
%     ba(ind) = net.simulate(k);
%     ind = ind + 1;
% end

% --------------------------------------
% goodness of fit
% --------------------------------------
nnfw.goodnessOfFit(toolbox, t, 'mse')
nnfw.Util.CostFunction.MSE.f(toolbox, t)

nnfw.goodnessOfFit(ba, t, 'mse')
nnfw.Util.CostFunction.MSE.f(ba, t)

nnfw.goodnessOfFit(ba, toolbox, 'mse')
nnfw.Util.CostFunction.MSE.f(ba, toolbox)

% --------------------------------------
% plot
% --------------------------------------
figure(2);
hold on
plot(t, 'r'); % target
plot(toolbox, 'k'); % toolbox
plot(ba, 'g'); % ba
legend('target','toobox','ba');
hold off