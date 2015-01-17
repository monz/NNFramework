%% Bremse Moment

%% load data
loadData;

%% define input and target data
x = Vertikal_Kraft';
t = Vertikal_DeltaP';

%% define, train and simulate neural network
y = con2seq(t);
u = con2seq(x);

d1 = [1:3];
d2 = [1:2];
narx_net = narxnet(d1,d2,20);
[pr,Pi,Ai,tr] = preparets(narx_net,u,{},y);

narx_net = train(narx_net,pr,tr,Pi);
yp = sim(narx_net,pr,Pi);

simVal = cell2mat(yp);
tarVal = cell2mat(tr);

%% plot results
figure(2)
plot(tarVal,'r')
hold on
plot(simVal,'g')
legend('Bremse Moment','FNN')

%% rate results
FIT = nnfw.goodnessOfFit(simVal',tarVal','NRMSE')