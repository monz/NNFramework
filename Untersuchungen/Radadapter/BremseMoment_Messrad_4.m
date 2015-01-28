%% Ansteuerung fuer Bremse Moment am Messrad

clear;
close all;
%% settings

% neural net settings
numNeurons = 10;
maxIter = 50;
delay1 = 1:20;
useToolbox = true;
delayNet = false;
flipTime = false;

% plot settings
figureNr = 2;
plotValidateData = false;

% data settings
useVertikalDeltaP = true;

%% load data

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';
validateFile = '03_APRBS_APK1_sim3_DRV_100Pct_RSP.tim';

trdata = loadDataF(trainFile);
valdata = loadDataF(validateFile);

%% define input and target data

if useVertikalDeltaP
    p = [trdata.Longitudinal_DeltaP'; trdata.Bremse_LVDT'; trdata.Vertikal_DeltaP'; trdata.Bremse_DeltaP'];
    pVal = [valdata.Longitudinal_DeltaP'; valdata.Bremse_LVDT'; valdata.Vertikal_DeltaP'; valdata.Bremse_DeltaP'];
else
    p = [trdata.Longitudinal_DeltaP'; trdata.Vertikal_DeltaP'; trdata.Bremse_DeltaP'];
    pVal = [valdata.Longitudinal_DeltaP'; valdata.Vertikal_DeltaP'; valdata.Bremse_DeltaP'];
end
t = [trdata.MYMR'];
tVal = [valdata.MYMR'];


%% define and train neural network

% flip left right
if delayNet
    if flipTime
        p = fliplr(p);
        t = fliplr(t);
        if plotValidateData
            pVal = fliplr(pVal);
            tVal = fliplr(tVal);
        end
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
    
    if plotValidateData
        yVal = con2seq(tVal);
        uVal = con2seq(pVal);
        [prVal,PiVal,~,trVal] = preparets(net,uVal,yVal);
        
        yVal = net(prVal, PiVal);
        tVal = cell2mat(trVal);
        yVal = cell2mat(yVal);
        if flipTime
            yVal = fliplr(yVal);
            tVal = fliplr(tVal);
        end
    end
    
    % reverse flip left right
    if flipTime
        y = fliplr(y);
        t = fliplr(t);
    end
end

if useToolbox && ~delayNet
    y = net(p);
    
    if plotValidateData
        yVal = net(pVal);
    end
elseif ~delayNet
    y = net.simulate(p);
    
    if plotValidateData
        yVal = net.simulate(pVal);
    end
end

%% rate results
fit = nnfw.goodnessOfFit(y',t','NRMSE')
if plotValidateData
    fitVal = nnfw.goodnessOfFit(yVal',tVal','NRMSE')
end

%% plot results
figure(figureNr)
hold on
    title('Ansteuerung Bremse Moment');
    
    sp(1) = subplot(511);
    plot(p(1,:),'r')
    legend('Longitudinal DeltaP')
    sp(2) = subplot(512);
    plot(p(2,:),'r')
    legend('Bremse LVDT')
    if useVertikalDeltaP
        sp(3) = subplot(513);
        plot(p(3,:),'r')
        legend('Vertikal DeltaP');
        sp(4) = subplot(514);
        plot(p(4,:),'r')
        legend('Bremse DeltaP');
    else
        sp(4) = subplot(514);
        plot(p(3,:),'r')
        legend('Bremse DeltaP');
    end
    sp(5) = subplot(515);
    plot(t,'r');
    hold on
    plot(y,'g');
    
    if plotValidateData
        hold on
        plot(tVal,'c');
        hold on
        plot(yVal,'k');
        legend('Bremse Moment Messrad','ANN','Target Validation', 'ANN Validation');
    else
        legend('Bremse Moment Messrad','ANN');
    end
    
    linkaxes(sp,'x');
hold off