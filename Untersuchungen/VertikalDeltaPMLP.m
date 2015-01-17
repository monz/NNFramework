%% Vertikal DeltaP

%% load data
loadData;

%% define input and target data
x = Vertikal_Kraft';
t = Vertikal_DeltaP';

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
legend('Vertikal DeltaP','FNN')

%% rate results
FIT = nnfw.goodnessOfFit(y',t','NRMSE')