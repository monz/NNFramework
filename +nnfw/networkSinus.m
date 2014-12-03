clear;
clc;

% --------------------------------------
% init training values
% --------------------------------------
% p = (-2:.1:2);
p = (-5:.1:5);
t = cos(pi*p/2);

% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
% net1 = feedforwardnet(5);
% net1 = train(net1,p,t);
% y_d = net1(p);

% --------------------------------------
% init nn-framework
% --------------------------------------
net = nnfw.FeedForward(5);
net.configure(p,t);
net.optim.abortThreshold = 1e-5;
% net.optim.maxIter = 10;
% net.layers{1}.f = nnfw.Util.Activation.LOGSIG;

% --------------------------------------
% train network
% --------------------------------------
[E, g, output, lambda, jacobian] = net.train(p,t);
y_d = net.simulate(p);

% --------------------------------------
% plot
% --------------------------------------
figure(2);
hold on

plot(t, 'r'); % target
plot(y_d, 'g'); % ba
% plot(toolbox, 'k'); % toolbox
% legend('target','toobox','ba');
legend('target','ba');

hold off