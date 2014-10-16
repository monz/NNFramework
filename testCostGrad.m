clear all;
clc;

% --------------------------------------
% calculate gradients
% --------------------------------------
% syms x y z q r s t p;
% g(x, y, z, q, r, s, t, p) = ((1+sin((pi/4)*p)) - (t + s*(logsig(y*p+q)) + r*(logsig(x*p+z)) ) )^2;
% diff(g, t)
% diff(g, z)
% diff(g, q)
% diff(g, x)
% diff(g, y)
% diff(g, r)
% diff(g, s)

% --------------------------------------
% init nn-framework
% --------------------------------------
p = (-2:0.1:2);
% p = 1;
target = 1 + sin((pi/4)*p);

IW = [-0.27; -0.41];
LW = [0.09 -0.17];
b1 = [-0.48; -0.13];
b2 = [0.48];

net = nnfw.FeedForward(1, 2, 1);
net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
% net.inputs{1}.size = 1;
net.layers{1}.size = 2;
% net.outputs{2}.size = 1;
net.IW{1} = IW;
net.LW{2,1} = LW;
net.b{1,1} = b1;
net.b{2,1} = b2;

[cost, grad] = net.train(p, target);
% sum(grad,1)
grad


% for k = 1:length(grad)
%    grad{k} 
% end
% for k = 1:length(bGrad)
%    bGrad{k} 
% end

% --------------------------------------
% prepare starting weights in numeric calculation
% --------------------------------------
x = -0.27; %x // w111
y = -0.41; %y // w211
z = -0.48; %z // b11
q = -0.13; %q // b21
r = 0.09; %r // w112
s = -0.17; %s // w122
t = 0.48; %t // b12
% --------------------------------------
% prepare gradient matrix
% --------------------------------------

grad_derivated = zeros(length(p), 7);

pIn = p;
for k = 1:length(pIn)
    p = pIn(k);
    grad_derivated(k, 1) = (2*p*r*exp(- z - p*x)*(t - sin((pi*p)/4) + s/(exp(- q - p*y) + 1) + r/(exp(- z - p*x) + 1) - 1))/(exp(- z - p*x) + 1)^2;
    grad_derivated(k, 2) = (2*p*s*exp(- q - p*y)*(t - sin((pi*p)/4) + s/(exp(- q - p*y) + 1) + r/(exp(- z - p*x) + 1) - 1))/(exp(- q - p*y) + 1)^2;
    grad_derivated(k, 3) = (2*r*exp(- z - p*x)*(t - sin((pi*p)/4) + s/(exp(- q - p*y) + 1) + r/(exp(- z - p*x) + 1) - 1))/(exp(- z - p*x) + 1)^2;
    grad_derivated(k, 4) = (2*s*exp(- q - p*y)*(t - sin((pi*p)/4) + s/(exp(- q - p*y) + 1) + r/(exp(- z - p*x) + 1) - 1))/(exp(- q - p*y) + 1)^2;

    grad_derivated(k, 5) = (2*(t - sin((pi*p)/4) + s/(exp(- q - p*y) + 1) + r/(exp(- z - p*x) + 1) - 1))/(exp(- z - p*x) + 1);
    grad_derivated(k, 6) = (2*(t - sin((pi*p)/4) + s/(exp(- q - p*y) + 1) + r/(exp(- z - p*x) + 1) - 1))/(exp(- q - p*y) + 1);
    grad_derivated(k, 7) = 2*t - 2*sin((pi*p)/4) + (2*s)/(exp(- q - p*y) + 1) + (2*r)/(exp(- z - p*x) + 1) - 2;
end
sum(grad_derivated, 1)

% 
% isequal(grad, grad_derivated)
% abs(grad - grad_derivated) < 32 * eps(max(grad, grad_derivated))
