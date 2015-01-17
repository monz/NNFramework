%% Lenkung Moment

%% load data
loadData;

%% define input and target data
x = [Longitudinal_Kraft'; Lateral1_KMD'; Lateral2_KMD'; Lateral3_KMD'; Lateral_Kraft'; Longitudinal_DeltaP'];
t = [MZMR'];

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
legend('Lenkung Moment','FNN')

%% rate results
FIT = nnfw.goodnessOfFit(y',t','NRMSE')