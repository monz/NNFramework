%% Lenkung Moment Reverse

%% load data
loadData;

%% define input and target data
x = [MZMR'];
t = [Longitudinal_Kraft'; Lateral1_KMD'; Lateral2_KMD'; Lateral3_KMD'; Lateral_Kraft'; Longitudinal_DeltaP'];
% t = [FXAK'; MXAK'; FYAK'; MZAK'];


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
subplot(411)
plot(tarVal(1,:),'r')
hold on
plot(simVal(1,:),'g')
legend('Longitudinal Kraft','FNN')
subplot(412)
plot(tarVal(2,:),'r')
hold on
plot(simVal(2,:),'g')
legend('Lateral1 KMD','FNN')
subplot(413)
plot(tarVal(5,:),'r')
hold on
plot(simVal(5,:),'g')
legend('Lateral Kraft','FNN')
subplot(414)
plot(tarVal(6,:),'r')
hold on
plot(simVal(6,:),'g')
legend('Longitudinal DeltaP','FNN')

%% rate results
FIT = nnfw.goodnessOfFit(simVal',tarVal','NRMSE')