%% Lenkung Moment

%% load data
loadData;

%% define input and target data
% x = [FXAK'; MXAK'; FYAK'; MZAK'];
x = [Longitudinal_Kraft'; Lateral1_KMD'; Lateral2_KMD'; Lateral3_KMD'; Lateral_Kraft'; Longitudinal_DeltaP'];
t = [MZMR'];

%% define, train and simulate neural network
y = con2seq(t);
u = con2seq(x);

d1 = [1:2];
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
legend('Lenkung Moment','FNN')

%% rate results
FIT = nnfw.goodnessOfFit(simVal',tarVal','NRMSE')