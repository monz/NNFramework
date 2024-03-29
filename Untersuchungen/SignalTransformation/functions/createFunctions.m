%%% -----------------------------------------------------------------------
%   this script creates simple test signal cluster and saves the generated
%   data to a file, because of the random data. This file can be loaded
%   repeatedly to study different neural network architectures.
%%% -----------------------------------------------------------------------

clear;
close all;

%% settings

filename = 'noiseFastSinOffset.mat';

range = -2:0.03:2;
noiseAmplitude = 0.32;

pClusterWidth = -0.3:0.02:0.1;
tClusterWidth = 0.5:0.02:0.9;

%% create signal cluster

numInputs = length(pClusterWidth);
p = zeros(numInputs, length(range));
myRand = rand;
for k = 1:numInputs
    p(k,:) = sin(pClusterWidth(k)+pi*range*myRand)+rand;
    p(k,:) = p(k,:) + rand(1,length(p(k,:)))*noiseAmplitude;
end

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

%% save to file

save(filename,'p','t','range');