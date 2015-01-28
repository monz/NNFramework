%% Ansteuerung fuer Bremse Moment Reverse

clear;
close all;
%% settings

% neural net settings
numNeurons = [10];
maxIter = 50;
delay1 = 1:50;
useToolbox = true;
delayNet = true;
flipTime = false;

% plot settings
figureNr = 2;

%% load data

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
    plot(p,'r')
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