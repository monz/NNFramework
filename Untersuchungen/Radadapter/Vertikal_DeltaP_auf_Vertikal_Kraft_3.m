%% Vertikal_DeltaP auf Vertikal_Kraft

clear;
close all;
%% settings

% neural net settings
numNeurons = [10];
maxIter = 50;
useToolbox = true;
delayNet = false;
flipTime = false;
delay1 = 1:10;

% plot settings
figureNr = 2;
plotValidateData = false;

%% load data

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';
validateFile = '02_APRBS_APK1_sim2_DRV_100Pct_RSP.tim';

traindata = loadDataF(trainFile);
validatedata = loadDataF(validateFile);

%% define input and target data

p = [traindata.Vertikal_DeltaP'];
% p = p - p(1);
t = [traindata.Vertikal_Kraft'];
t = (t + 1.4)/(0.1*0.001*3978);
% t = t - t(1);
pVal = [validatedata.Longitudinal_DeltaP'];
tVal = [validatedata.Longitudinal_Kraft'];

%% define and train neural network

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
    if delayNet
        yVal = con2seq(tVal);
        uVal = con2seq(pVal);

        [prVal,PiVal,~,tr] = preparets(net,uVal,yVal);
        yVal = net(prVal,PiVal);

        tVal = cell2mat(tr);
        yVal = cell2mat(yVal);

        % reverse flip left right
        if flipTime
            yVal = fliplr(yVal);
            tVal = fliplr(tVal);
        end
    end
    
    if useToolbox && ~delayNet
        yVal = net(pVal);
    elseif ~delayNet
        yVal = net.simulate(pVal);
    end
end

%% plot results
figure(figureNr)
hold on
    title('Longidutinal DeltaP auf Longitudinal Kraft');
    plot(t,'r');
    plot(y,'g');
    if plotValidateData
        plot(tVal,'c');
        plot(yVal,'k');
        legend('Vertikal Kraft','ANN','Target Validation', 'ANN Validation');
    else
        legend('Vertikal Kraft','ANN');
    end
hold off

%% rate results
fit = nnfw.goodnessOfFit(y',t','NRMSE')
if plotValidateData
    if delayNet
        fitVal = nnfw.goodnessOfFit(yVal',tVal','NRMSE')
    else
        fitVal = nnfw.goodnessOfFit(yVal',tVal','NRMSE')
    end
end