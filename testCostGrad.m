clear all;
clc;
% --------------------------------------
% init nn-framework
% --------------------------------------
p = 2;
target = 1 + sin((pi/4)*p);

IW = [-0.27; -0.41];
LW = [0.09 -0.17];
b1 = [-0.48; -0.13];
b2 = [0.48];

net = nnfw.FeedForward(1, 2, 1);
net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
net.layers{1}.size = 2;
net.IW{1} = IW;
net.LW{2,1} = LW;
net.b{1,1} = b1;
net.b{2,1} = b2;

[cost, grad, bGrad] = net.train(p, target);