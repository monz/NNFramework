% block train, the network memorizes the target values

clear;
close all;

%% choose parameter

numNeurons = 2;
useToolbox = true;
useMinmaxMapping = false;

%% prepare values

range = -2:0.1:2;
t = sin(pi*range/2)+0.5; % reference curve

% buld curves with offset
sampleRange = 2:0.2:2.8;
t2 = zeros(length(sampleRange), length(t));
for k = 1:length(sampleRange)
    t2(k,:) = t+sampleRange(k);
end

outerData = t+1.5; % create data beyond training data

figure(1)
hold on
    rfData = plot(range, t, 'LineWidth', 2); % reference curve
    coData = plot(range, t2,'g'); % data to convert to reference
hold off

numInputs = size(t2,1);
inputLength = length(t2);

p = t2'; % transpose input values, net input is a entire curve
t = repmat(t', 1, numInputs); % repeat output, for each input curve one output curve, always the same

% change every second reference curve by smalles possible amount
% ind = 2:2:numInputs;
% ind = 1;
% t(:,ind) = t(:,ind) + eps(t(:,ind)); % add smalles possible amount, leads to expected behavior with nn-toolbox

%% train net

if useToolbox
    net = feedforwardnet(numNeurons);
    if ~useMinmaxMapping
        net.inputs{1}.processFcns = {};
        net.outputs{2}.processFcns = {};
    end
    net.trainParam.epochs = 50;
    net = train(net, p, t);
else
    net = nnfw.FeedForward(numNeurons);
    net.configure(p,t);
    net.optim.maxIter = 50;
    net.optim.minmaxMapping = useMinmaxMapping;
    net.optim.abortThreshold = 1e-32;
    net.train(p,t);
end

%% simulate

if useToolbox
    y = net(p);
    yRand = net(rand(inputLength, 1)*2);
    yOuter = net(outerData');
else
    y = net.simulate(p);
    yRand = net.simulate(rand(inputLength, 1)*2);
    yOuter = net.simulate(outerData');
end

%% plot

figure(1)
hold on
    for k = 1:numInputs
        trData = plot(range, y(:, k)', 'r');
    end
    rnData = plot(range, yRand','c');
    ouData = plot(range, yOuter','m');
    legend([rfData, coData(1), trData, rnData, ouData], 'ReferenceData', 'DataToConvert', 'DataConverted', 'RandomData', 'BeyondSpecsData');
hold off