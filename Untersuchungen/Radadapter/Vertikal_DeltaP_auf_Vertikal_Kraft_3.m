%%% -----------------------------------------------------------------------
% =========================================================================
%   Wheel behavior Study Script
%   Train Method: SISO with Feedforward/Delay-Net
%
%   Vertikal_DeltaP to Vertikal_Kraft
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
%   plotValidateData:   if true the validation data gets simulated with the
%                       trained network and plotted to see how good the
%                       neural network is for generalization
%
%   figureNr:           sets the figure handle
%
%   trainFile:          name of file containing the training data
%
%   validateFile:       name of file containing the validation data
%%% -----------------------------------------------------------------------

clear;
close all;
%% set options

% neural net settings
numNeurons = [3];
maxIter = 50;
useToolbox = false;
delayNet = false;
flipTime = false;
delay1 = 1:2;

% plot settings
figureNr = 2;
plotValidateData = false;

% load data settings

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';
validateFile = '02_APRBS_APK1_sim2_DRV_100Pct_RSP.tim';

traindata = loadDataF(trainFile);
validatedata = loadDataF(validateFile);

%% define input and target data

p = [traindata.Vertikal_DeltaP'];
% p = p - p(1);
% p = p(1:end-1);
t = [traindata.Vertikal_Kraft'];
% t = [traindata.FZMR'];
% t = t(2:end);
% t = (t + 1.4)/(0.1*0.001*3978);
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
    title('Vertical DeltaP to Vertical Force');
    xlabel('Vertical DeltaP [bar]');
    ylabel('Simulated Model Output; Vertical Force [kN]');
%     set(gca, 'XTick', 49700:0.1:49900);
    grid on;
    plot(t,'r');
    plot(y,'g');
%     plot(49700:49900,t(49700:49900),'r');
%     plot(49700:49900,y(49700:49900),'g');
    if plotValidateData
        plot(tVal,'c');
        plot(yVal,'k');
        legend('Vertical Force','ANN','Target Validation', 'ANN Validation');
    else
        legend('Vertical Force','ANN Simulated Force');
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