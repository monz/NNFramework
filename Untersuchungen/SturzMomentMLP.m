%% Sturz Moment

%% load data
loadData;

%% define input and target data
x = [MXAK'; FZAK'];
t = [MXMR'];

%% define, train and simulate neural network
net = feedforwardnet(10);
% net.trainParam.epochs = 50;

net = train(net,x,t);
y = net(x);

% net = nnfw.FeedForward(10);
% net.configure(x,t);
% net.optim.maxIter = 10;
% net.train(x,t);
% y = net.simulate(x);

%% plot results
figure(2)
plot(t,'r')
hold on
plot(y,'g')
legend('Bremse Moment','FNN')

%% rate results
FIT = nnfw.goodnessOfFit(y',t','NRMSE')