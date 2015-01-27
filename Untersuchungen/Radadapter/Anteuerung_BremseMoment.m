%% Ansteuerung für Bremse Moment

clear;
close all;
%% settings

% neural net settings
numNeurons = 10;
maxIter = 50;
delay1 = 1:10;
useToolbox = true;
delayNet = false;
flipTime = false;

% plot settings
figureNr = 2;
plotValidateData = false;

%% load data

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';
validateFile = '03_APRBS_APK1_sim3_DRV_100Pct_RSP.tim';

trdata = loadDataF(trainFile);
valdata = loadDataF(validateFile);

%% define input and target data

p = [trdata.Bremse_Moment'];
t = [trdata.Longitudinal_DeltaP'; trdata.Bremse_DeltaP'; trdata.Bremse_LVDT'];

% xV = [valdata.Longitudinal_DeltaP'];
% tV = [valdata.Longitudinal_Kraft'];

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

if plotValidateData
    if useToolbox
        yVal = net(xV);
    else
        yVal = net.simulate(xV);
    end
end

%% rate results
fit = nnfw.goodnessOfFit(y',t','NRMSE')
if plotValidateData
    fitVal = nnfw.goodnessOfFit(yVal',tV','NRMSE')
end

%% plot results
figure(figureNr)
hold on
    title('Ansteuerung Bremse Moment');
    subplot(411)
    plot(t(1,:),'r')
    hold on
    plot(y(1,:),'g')
    legend('Target 1','ANN')
    subplot(412)
    plot(t(2,:),'r')
    hold on
    plot(y(2,:),'g')
    legend('Target 2','ANN')
    subplot(413)
    plot(t(3,:),'r')
    hold on
    plot(y(3,:),'g')
    legend('Target 3','ANN');
    
    if plotValidateData
        plot(tV,'c');
        plot(yVal,'k');
        legend('Target','ANN','Target Validation', 'ANN Validation');
    end
hold off