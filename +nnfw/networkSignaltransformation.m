clear;
clc;
close all;

% load data from .mat-file:
load('clusteringData');

% select test part:
idPtidC = 143;

% kt2:
dataindexKT2 = ismember(clusteringData(idPtidC).testbench, 'kt2');
% plot(clusteringData(idPtidC).xData(:,dataindexKT2), clusteringData(idPtidC).yData(:,dataindexKT2), 'Color', [0.0 0.0 0.7]);

% kt3:
dataindexKT3 = ismember(clusteringData(idPtidC).testbench, 'kt3');
% plot(clusteringData(idPtidC).xData(:,dataindexKT3), clusteringData(idPtidC).yData(:,dataindexKT3), 'Color', [0.0 0.7 0.0]);

% p = clusteringData(idPtidC).xData(:,dataindexKT2)';
% t = clusteringData(idPtidC).yData(:,dataindexKT2)';
p = clusteringData(idPtidC).yData(:,2)';
t = clusteringData(idPtidC).yData(:,1)';

% toolbox
% net = feedforwardnet(15);
% net.trainParam.epochs = 50;
% net = train(net, p,t);
% y = net(p);

% nnfw
net = nnfw.FeedForward([10 10]);
net.configure(p,t);
net.optim.maxIter = 100;
net.train(p,t);
y = net.simulate(p);

% plot
figure(2);
hold on
plot(p,y);
% plot(p',y','r');
% plot(p',t','g');
hold off;
