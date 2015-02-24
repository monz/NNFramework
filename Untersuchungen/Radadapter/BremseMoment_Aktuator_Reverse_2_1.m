%%% -----------------------------------------------------------------------
% =========================================================================
%   Wheel behavior Study Script
%   Train Method: SISO with Feedforward/Delay-Net
%
%   Activation for Bremse Moment on Actuator in Reverse Order
% =========================================================================
%
% #########################################################################
%   DO NOT CHANGE CODE BELOW THE "set options" SECTION
% #########################################################################
%
%   Possible Settings:
%   ------------------
%
%   numNeurons:         sets the number of neurons and layers in the neural network
%                       e.g. single hidden layer with 10 neurons; 10
%                       or two hidden layer with each 10 neurons; [10 10]
%
%   maxIter:            sets the max number of training iterations
%                       - positive number - e.g. 50
%
%   useToolbox:         if true uses MATLAB-NNToolbox for training, if false it
%                       uses the NN-Framework - except for DelayNet
%
%   delayNet:           if true use delayNet for training, if false switch
%                       to NARX-Net
%
%   flipTime:           if true the input and target values get flipped from
%                       left to right, using the fliplr MATLAB method.
%
%   delay1:             sets the input delay (for Delay- and NARX-Net)
%                       e.g. 1:2 or 1:10
%
%   figureNr:           sets the figure handle
%
%   trainFile:          name of file containing the training data
%%% -----------------------------------------------------------------------

clear;
close all;

%% set options

% neural net settings
numNeurons = [10];
maxIter = 50;
delay1 = 1:50;
useToolbox = true;
delayNet = false;
flipTime = false;

% plot settings
figureNr = 2;

% load data settings

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';

trdata = loadDataF(trainFile);

%% define input and target data

p = [trdata.Bremse_Moment'];
t = [trdata.Longitudinal_DeltaP'; trdata.Bremse_DeltaP'; trdata.Bremse_LVDT'];

%% define and train neural network

% flip left right
if delayNet
    if flipTime
        p = fliplr(p);
        t = fliplr(t);
    end
    y = con2seq(t);
    u = con2seq(p);

    net = timedelaynet(delay1,numNeurons);
    net.trainParam.epochs = maxIter;
    [pr,Pi,Ai,tr,~,shift] = preparets(net,u,y);
    net = train(net,pr,tr,Pi,Ai);
end

if useToolbox && ~delayNet
    net = feedforwardnet(numNeurons);
    net.trainParam.epochs = maxIter;
    net = train(net,p,t);
elseif ~delayNet
    net = nnfw.FeedForward(numNeurons);
    net.configure(p,t);
    net.optim.maxIter = maxIter;
    net.train(p,t);
end

%% simulate data
if delayNet
    y = net(pr,Pi);

    t = cell2mat(tr);
    y = cell2mat(y);
    
    % reverse flip left right
    if flipTime
        y = fliplr(y);
        t = fliplr(t);
    end
end

if useToolbox && ~delayNet
    y = net(p);
elseif ~delayNet
    y = net.simulate(p);
end

%% rate results
fit = nnfw.goodnessOfFit(y',t','NRMSE')

%% plot results
figure(figureNr)
hold on
    title('Ansteuerung Bremse Moment Aktuator Reverse');
    sp(1) = subplot(411);
    plot(p(1,:),'r')
    legend('BremseMoment');
    sp(2) = subplot(412);
    plot(t(1,:),'r')
    hold on
    plot(y(1,:),'g')
    legend('Longitudinal DeltaP','ANN')
    sp(3) = subplot(413);
    plot(t(2,:),'r')
    hold on
    plot(y(2,:),'g')
    legend('Bremse DeltaP','ANN')
    sp(4) = subplot(414);
    plot(t(3,:),'r')
    hold on
    plot(y(3,:),'g')
    legend('Bremse LVDT','ANN');

    linkaxes(sp,'x');
hold off