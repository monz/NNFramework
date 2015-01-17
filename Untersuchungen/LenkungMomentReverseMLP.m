%% Lenkung Moment Reverse

%% load data
loadData;

%% define input and target data
x = [MZMR'];
t = [Longitudinal_Kraft'; Lateral1_KMD'; Lateral2_KMD'; Lateral3_KMD'; Lateral_Kraft'; Longitudinal_DeltaP'];

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
subplot(411)
plot(t(1,:),'r')
hold on
plot(y(1,:),'g')
legend('Longitudinal Kraft','FNN')
subplot(412)
plot(t(2,:),'r')
hold on
plot(y(2,:),'g')
legend('Lateral1 KMD','FNN')
subplot(413)
plot(t(5,:),'r')
hold on
plot(y(5,:),'g')
legend('Lateral Kraft','FNN')
subplot(414)
plot(t(6,:),'r')
hold on
plot(y(6,:),'g')
legend('Longitudinal DeltaP','FNN')

%% rate results
FIT = nnfw.goodnessOfFit(y',t','NRMSE')