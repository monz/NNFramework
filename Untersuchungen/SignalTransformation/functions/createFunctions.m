clear;
close all;

range = -2:0.03:2;
noiseAmplitude = 0.32;

pClusterWidth = -0.3:0.02:0.1;
numInputs = length(pClusterWidth);
p = zeros(numInputs, length(range));
myRand = rand;
for k = 1:numInputs
    p(k,:) = sin(pClusterWidth(k)+pi*range*myRand)+rand;
    p(k,:) = p(k,:) + rand(1,length(p(k,:)))*noiseAmplitude;
end

tClusterWidth = 0.5:0.02:0.9;
numTargets = length(tClusterWidth);
myRand = rand;
for k = 1:numTargets
    t(k,:) = sin(tClusterWidth(k)+pi*range*myRand)+rand; % reference
    t(k,:) = t(k,:) + rand(1,length(t(k,:)))*noiseAmplitude;
end

hold on;
for k = 1:numInputs
    plot(range,p(k,:), 'r');
    plot(range,t(k,:), 'g');
end
hold off

t = t';
p = p';

% save('noiseFastSinOffset.mat','p','t','range');