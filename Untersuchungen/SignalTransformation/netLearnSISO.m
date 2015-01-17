% train SISO

clear;
close all;
%% choose parameter

useToolbox = true;
numNeurons = 2;

%% perpare values

range = -2:0.1:2;
t = sin(pi*range/2)+0.5; % reference curve

% build curves with offset
sampleRange = 2:0.2:2.8;
t2 = zeros(length(sampleRange), length(t));
for k = 1:length(sampleRange)
    t2(k,:) = t+sampleRange(k);
end

outerData = t+1.5; % create data beyond training data

figure(1)
hold on
    rfData = plot(range, t, 'LineWidth', 2); % reference curve, e.g. mean value
    coData = plot(range, t2,'g'); % data to convert to reference
hold off

numInputs = size(t2,1);
inputLength = length(t2);

p = reshape(t2', 1, inputLength*numInputs); % put all curves in ONE row, one after another
t = repmat(t, 1, numInputs); % repeat output curve

%% train net

if useToolbox
    net = feedforwardnet(numNeurons);
    net.trainParam.epochs = 50;
    net = train(net, p, t);
else
    net = nnfw.FeedForward(numNeurons);
    net.configure(p,t);
    net.optim.maxIter = 50;
    net.train(p,t);
end

%% simulate

if useToolbox
    y = net(p);
    yRand = net(rand(1,inputLength)*2);
    yOuter = net(outerData);
else
    y = net.simulate(p);
    yRand = net.simulate(rand(1,inputLength)*2);
    yOuter = net.simulate(outerData);
end

%% plot

figure(1)
hold on
    for k = 1:numInputs
        trData = plot(range, y((k-1)*inputLength+1:k*inputLength), 'r');
    end
    rnData = plot(range, yRand,'c');
    ouData = plot(range, yOuter,'m');
    legend([rfData, coData(1), trData, rnData, ouData], 'ReferenceData', 'DataToConvert', 'DataConverted', 'RandomData', 'BeyondSpecsData');
hold off