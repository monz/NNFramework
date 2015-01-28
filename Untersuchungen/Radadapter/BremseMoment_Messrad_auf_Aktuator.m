%% testo

clear;
close all;
%% settings

% neural net settings
numNeurons = [10];
maxIter = 50;
delay1 = 1:20;
useToolbox = true;
delayNet = true;
flipTime = false;

% plot settings
figureNr = 2;

%% load data

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';

trdata = loadDataF(trainFile);

%% define input and target data

p = [trdata.MYMR'];
t = [trdata.Bremse_Moment'];

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
    title('Ansteuerung Bremse Moment Messrad Reverse');
    plot(p,'c');
    plot(t,'r');
    plot(y,'g');
    legend('input', 'target', 'ann');
hold off