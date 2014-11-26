clear;
clc;

% --------------------------------------
% init training values
% --------------------------------------
p = [-2:.1:2; -2:.1:2];
t = [sin(pi*p(1,:)/2); cos(pi*p(2,:)/2)];

% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
% net1 = feedforwardnet(10);
% net1 = train(net1,p,t);
% y_d = net1(p);

% --------------------------------------
% init nn-framework
% --------------------------------------
net = nnfw.FeedForward(5);
net.configure(p,t);
net.layers{2}.f = nnfw.Util.Activation.LOGSIG;

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

plot(t(1,:), 'r'); % target
plot(t(2,:), 'b'); % target
plot(y_d(1,:), 'y'); % ba
plot(y_d(2,:), 'g'); % ba
legend('target sin','target cos', 'y_d sin', 'y_d cos');

hold off