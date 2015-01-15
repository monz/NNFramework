clear;
close all;

%% prepare data
% load signal transformation data
sigTrans_loadData;

% select test part
idPtidC = 143;

% kt2
dataindexKT2 = ismember(clusteringData(idPtidC).testbench, 'kt2');
% kt3
dataindexKT3 = ismember(clusteringData(idPtidC).testbench, 'kt3');

yMeanKT3 = mean(clusteringData(idPtidC).yData(:, dataindexKT3), 2);

p = clusteringData(idPtidC).yData(:,dataindexKT2)';
t = yMeanKT3';

% choose test data
testIndexes = [43 44 45 46 49 50 51 52 53];
numTest = length(testIndexes);
testDataInput = p(testIndexes, :);
p = p(~ismember(1:size(p,1),testIndexes), :);

%% train network
% toolbox
% net = feedforwardnet(5);
% net.trainParam.epochs = 30;
% net = train(net,p,t);
% y = net(p);

% nnfw
net = nnfw.FeedForward(5);
net.configure(p,t);
net.optim.maxIter = 40;
net.train(p,t);
y = net.simulate(p);

fit = nnfw.goodnessOfFit(y', t', 'NRMSE')

%% extrapolation

testDataSimulate = zeros(size(testDataInput));
fitTest = zeros(1, numTest);
dbTest = zeros(1, numTest);
for k = 1:size(testDataInput,1)
    testDataSimulate(k,:) = net.simulate(testDataInput(k,:));
%     testDataSimulate(k,:) = net(testDataInput(k,:));
    fitTest(k) = nnfw.goodnessOfFit(testDataSimulate(k,:)', t', 'NRMSE');
    dbTest(k) = daviesBouldin(testDataSimulate(k,:), t);
end

% mean fitTest; dbTest
meanFitTest = mean(fitTest)
meanDbTest = mean(dbTest)

%% plot
figure(2);
hold on
    title('Oelpumpe KT2 auf KT3');
    xlabel(clusteringData(idPtidC).xLabel);
    ylabel(clusteringData(idPtidC).yLabel);
    % plot(p',y','r'); % Verhältnis Eingang, Simulation
    plot(clusteringData(idPtidC).xData(:,dataindexKT3)', y, 'g', 'LineWidth', 2);
    %plot mean data
    plot(clusteringData(idPtidC).xData(:,dataindexKT3)', yMeanKT3', '--r');
    %plot extrapolation
    for k = 1:size(testDataInput,1)
        plot(clusteringData(idPtidC).xData(:,dataindexKT3)', testDataSimulate(k,:), 'r');
    end
    
    legend('ANN', 'KT3 Mean', 'extrapolation');
hold off;
plot_kt2_to_kt3(idPtidC);