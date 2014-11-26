clear;
clc;

% --------------------------------------
% init training values
% --------------------------------------
[p,t] = simpleclass_dataset;

% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
net1 = feedforwardnet(10);
net1 = train(net1,p,t);
y_d = net1(p);

% --------------------------------------
% init nn-framework
% --------------------------------------
net = nnfw.FeedForward(10);
net.configure(p,t);
net.layers{2}.f = nnfw.Util.Activation.LOGSIG;

% --------------------------------------
% train network
% --------------------------------------
% [E, ~, output, lambda, jacobian] = net.train(p,t);
% y_d = net.simulate(p);
% map outputs to 0...1
y_d2 = zeros(size(y_d));
% extract indexes of max value of each column
[~, i] = max(y_d, [], 1);
% map max values to given value
y_d2(sub2ind(size(y_d), i, 1:length(i))) = 1;

% --------------------------------------
% plot
% --------------------------------------
figure(2);
hold on

plotconfusion(t,y_d2);

hold off