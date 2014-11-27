clear;
clc;

% --------------------------------------
% init training values
% --------------------------------------
% p = [-2:.1:2; -2:.1:2];
p = [0.9; 0.9];
t = [sin(pi*p(1,:)/2); cos(pi*p(2,:)/2)];

% --------------------------------------
% init/train nn-toolbox
% --------------------------------------
% net1 = feedforwardnet(4);
% net1 = train(net1,p,t);
% y_d = net1(p);

% --------------------------------------
% init nn-framework
% --------------------------------------
net = nnfw.FeedForward(2);
net.configure(p,t);
net.setWeights([0.75 0.69 -0.54 0.17 -0.23 0.47 0.18 0.71 0.38 0.72 0.11 0.33]');
net.simulate(p, false);
costFcn = net.makeCostFcn2(@nnfw.Util.componentError, p, t);
[F, J] = costFcn(net.getWeightVector());

% --------------------------------------
% train network
% --------------------------------------
% [E, ~, output, lambda, jacobian] = net.train(p,t);
% y_d = net.simulate(p);

% --------------------------------------
% plot
% --------------------------------------
% figure(2);
% hold on
% 
% plot(t(1,:), 'r'); % target
% plot(t(2,:), 'b'); % target
% plot(y_d(1,:), 'y'); % ba
% plot(y_d(2,:), 'g'); % ba
% legend('target sin','target cos', 'y_d sin', 'y_d cos');
% 
% hold off