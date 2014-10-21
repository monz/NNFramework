clear all;
clc;

load bodyfat_dataset
% --------------------------------------
% calculate gradients
% --------------------------------------
% syms x y z q r s t p target;
% g = (target - (t + s*(logsig(y*p+q)) + r*(logsig(x*p+z)) ) )^2;
% diff(g, x)
% diff(g, y)
% diff(g, z)
% diff(g, q)
% diff(g, r)
% diff(g, s)
% diff(g, t)

% syms w1011 w1021 w1031 w1041 w1051 w1061 w1071 w1081 w1091 w1101 w1111 w1121 w1131 w2011 w2021 w2031 w2041 w2051 w2061 w2071 w2081 w2091 w2101 w2111 w2121 w2131 b11 b21 w112 w122 b12 p_1 p_2 p_3 p_4 p_5 p_6 p_7 p_8 p_9 p_10 p_11 p_12 p_13 target;
% gradss = (target - (b12 + w122*(logsig(w2011*p_1 + w2021*p_2 + w2031*p_3 + w2041*p_4 + w2051*p_5 + w2061*p_6 + w2071*p_7 + w2081*p_8 + w2091*p_9 + w2101*p_10 + w2111*p_11 + w2121*p_12 + w2131*p_13 + b21)) + w112*(logsig(w1011*p_1 + w1021*p_2 + w1031*p_3 + w1041*p_4 + w1051*p_5 + w1061*p_6 + w1071*p_7 + w1081*p_8 + w1091*p_9 + w1101*p_10 + w1111*p_11 + w1121*p_12 + w1131*p_13 + b11)) ) )^2;

% diff(gradss, w1011)
% diff(gradss, w1021)
% diff(gradss, w1031)
% diff(gradss, w1041)
% diff(gradss, w1051)
% diff(gradss, w1061)
% diff(gradss, w1071)
% diff(gradss, w1081)
% diff(gradss, w1091)
% diff(gradss, w1101)
% diff(gradss, w1111)
% diff(gradss, w1121)
% diff(gradss, w1131)

% diff(gradss, w2011)
% diff(gradss, w2021)
% diff(gradss, w2031)
% diff(gradss, w2041)
% diff(gradss, w2051)
% diff(gradss, w2061)
% diff(gradss, w2071)
% diff(gradss, w2081)
% diff(gradss, w2091)
% diff(gradss, w2101)
% diff(gradss, w2111)
% diff(gradss, w2121)
% diff(gradss, w2131)
% 
% diff(gradss, b11)
% diff(gradss, b21)
% 
% diff(gradss, w112)
% diff(gradss, w122)
% diff(gradss, b12)

% --------------------------------------
% init weights
% --------------------------------------
w1011 = -0.27;
w1021 = 0.09;
w1031 = -0.13;
w1041 = -0.27;
w1051 = 0.09;
w1061 = -0.13;
w1071 = -0.41;
w1081 = -0.27;
w1091 = 0.09;
w1101 = -0.13;
w1111 = -0.27;
w1121 = 0.09;
w1131 = -0.13;

w2011 = -0.27;
w2021 = 0.09;
w2031 = -0.13;
w2041 = -0.27;
w2051 = 0.09;
w2061 = -0.13;
w2071 = -0.27;
w2081 = 0.09;
w2091 = -0.13;
w2101 = -0.27;
w2111 = 0.09;
w2121 = -0.13;
w2131 = -0.27;

b11 = 0.09;
b21 = -0.13;

w112 = -0.27;
w122 = 0.09;
b12 = -0.13;

% w1011 = rand();
% w1021 = rand();
% w1031 = rand();
% w1041 = rand();
% w1051 = rand();
% w1061 = rand();
% w1071 = rand();
% w1081 = rand();
% w1091 = rand();
% w1101 = rand();
% w1111 = rand();
% w1121 = rand();
% w1131 = rand();
% 
% w2011 = rand();
% w2021 = rand();
% w2031 = rand();
% w2041 = rand();
% w2051 = rand();
% w2061 = rand();
% w2071 = rand();
% w2081 = rand();
% w2091 = rand();
% w2101 = rand();
% w2111 = rand();
% w2121 = rand();
% w2131 = rand();
% 
% b11 = rand();
% b21 = rand();
% 
% w112 = rand();
% w122 = rand();
% b12 = rand();

% --------------------------------------
% init nn-framework
% --------------------------------------
p = bodyfatInputs;
target = bodyfatTargets;

IW = [w1011 w1021 w1031 w1041 w1051 w1061 w1071 w1081 w1091 w1101 w1111 w1121 w1131;
    w2011 w2021 w2031 w2041 w2051 w2061 w2071 w2081 w2091 w2101 w2111 w2121 w2131];
LW = [w112 w122];
b1 = [b11; b21];
b2 = [b12];

net = nnfw.FeedForward(1, 2, 1);
net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
net.layers{1}.size = 2;
net.IW{1} = IW;
net.LW{2,1} = LW;
net.b{1,1} = b1;
net.b{2,1} = b2;

[cost, grad] = net.train(p, target);
% sum(grad,1)
grad

% --------------------------------------
% prepare starting weights in numeric calculation
% --------------------------------------
% x = -0.27; %x // w111
% y = -0.41; %y // w211
% z = -0.48; %z // b11
% q = -0.13; %q // b21
% r = 0.09; %r // w112
% s = -0.17; %s // w122
% t = 0.48; %t // b12


% --------------------------------------
% prepare gradient matrix
% --------------------------------------

grad_derivated = zeros(length(p), net.getNumWeights());
% 
Q = length(p);
pIn = p;
tIn = target;
for q = 1:Q
    target = tIn(q);
    p_1 = pIn(1,q);
    p_2 = pIn(2,q);
    p_3 = pIn(3,q);
    p_4 = pIn(4,q);
    p_5 = pIn(5,q);
    p_6 = pIn(6,q);
    p_7 = pIn(7,q);
    p_8 = pIn(8,q);
    p_9 = pIn(9,q);
    p_10 = pIn(10,q);
    p_11 = pIn(11,q);
    p_12 = pIn(12,q);
    p_13 = pIn(13,q);
    
    grad_derivated(q, 1) = (2*p_1*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 2) = (2*p_2*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 3) = (2*p_3*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 4) = (2*p_4*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 5) = (2*p_5*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 6) = (2*p_6*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 7) = (2*p_7*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 8) = (2*p_8*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 9) = (2*p_9*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 10) = (2*p_10*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 11) = (2*p_11*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 12) = (2*p_12*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 13) = (2*p_13*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    
    grad_derivated(q, 14) = (2*p_1*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 15) = (2*p_2*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 16) = (2*p_3*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 17) = (2*p_4*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 18) = (2*p_5*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 19) = (2*p_6*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 20) = (2*p_7*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 21) = (2*p_8*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 22) = (2*p_9*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 23) = (2*p_10*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 24) = (2*p_11*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 25) = (2*p_12*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    grad_derivated(q, 26) = (2*p_13*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
    
    grad_derivated(q, 27) = (2*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
    grad_derivated(q, 28) = (2*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;

    grad_derivated(q, 29) = (2*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1);
    grad_derivated(q, 30) = (2*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1);
    grad_derivated(q, 31) = 2*b12 - 2*target + (2*w112)/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + (2*w122)/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1);
    
end
sum(grad_derivated, 1)
