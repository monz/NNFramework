clear;
clc;

% --------------------------------------
% init training values
% --------------------------------------
[p,t] = simpleclass_dataset;

% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
% net1 = feedforwardnet(5);
% net1 = train(net1,p,t);
% y_d = net1(p);

% --------------------------------------
% init nn-framework
% --------------------------------------
net = nnfw.FeedForward(5, true);
net.configure(p,t);
% net.layers{2}.f = nnfw.Util.Activation.LOGSIG;

% --------------------------------------
% train network
% --------------------------------------
[E, ~, output, lambda, jacobian] = net.train(p,t);
y_d = net.simulate(p);

% --------------------------------------
% plot
% --------------------------------------
figure(2);
hold on

plotconfusion(t,y_d);

hold off