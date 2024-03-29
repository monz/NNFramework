%%% -----------------------------------------------------------------------
% =========================================================================
%   Wheel behavior Study Script
%   Train Method: SISO with Feedforward/Delay-Net
%
%   Activation for Bremse Moment on Measuring Wheel
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
%   useVertikalDeltaP:  if true the training is performed with Vertikal_DeltaP
%                       as additional data
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
numNeurons = [10];
maxIter = 50;
delay1 = 1:20;
useToolbox = true;
delayNet = false;
flipTime = false;

% plot settings
figureNr = 2;
plotValidateData = true;

% data settings
useVerticalDeltaP = false;

% load data settings

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';
validateFile = '03_APRBS_APK1_sim3_DRV_100Pct_RSP.tim';

trdata = loadDataF(trainFile);
valdata = loadDataF(validateFile);

%% define input and target data

if useVerticalDeltaP
    p = [trdata.Longitudinal_DeltaP'; trdata.Bremse_LVDT'; trdata.Vertikal_DeltaP'; trdata.Bremse_DeltaP'];
    pVal = [valdata.Longitudinal_DeltaP'; valdata.Bremse_LVDT'; valdata.Vertikal_DeltaP'; valdata.Bremse_DeltaP'];
else
    p = [trdata.Longitudinal_DeltaP'; trdata.Bremse_LVDT'; trdata.Bremse_DeltaP'];
    pVal = [valdata.Longitudinal_DeltaP'; valdata.Vertikal_DeltaP'; valdata.Bremse_DeltaP'];
end
t = [trdata.MYMR'];
% t = [trdata.MYAK'];
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

    if useVerticalDeltaP
        sp(1) = subplot(511);
        plot(p(1,:),'r')
        xlabel('Frames [1/25 s]');
        ylabel('[Bar]');
        legend('Longitudinal DeltaP')
        
        sp(2) = subplot(512);
        plot(p(2,:),'r')
        xlabel('Frames [1/25 s]');
        ylabel('[mm]');
        legend('Brake LVDT')

        sp(3) = subplot(513);
        plot(p(3,:),'r')
        xlabel('Frames [1/25 s]');
        ylabel('[Bar]');
        legend('Vertical DeltaP');
        
        sp(4) = subplot(514);
        plot(p(4,:),'r')
        xlabel('Frames [1/25 s]');
        ylabel('[Bar]');
        legend('Brake DeltaP');

        sp(5) = subplot(515);
        plot(t,'r');
        hold on
        plot(y,'g');
    
        linkaxes(sp(1:4),'y');
    else
        sp(1) = subplot(411);
        plot(p(1,:),'r')
        xlabel('Frames [1/25 s]');
        ylabel('[Bar]');
        legend('Longitudinal DeltaP')
        
        sp(2) = subplot(412);
        plot(p(2,:),'r')
        xlabel('Frames [1/25 s]');
        ylabel('[mm]');
        legend('Brake LVDT')
        
        sp(3) = subplot(413);
        plot(p(3,:),'r')
        xlabel('Frames [1/25 s]');
        ylabel('[Bar]');
        legend('Brake DeltaP');
        
        sp(4) = subplot(414);
        plot(t,'r');
        hold on
        plot(y,'g');
    
        linkaxes(sp(1:3),'y');
    end
    
    if plotValidateData
        hold on
        plot(tVal,'c');
        hold on
        plot(yVal,'k');
        legend('Brake Torque Wheel Adapter','ANN','Target Validation', 'ANN Validation');
        hold off
    else
        legend('Brake Torque Wheel Adapter','ANN');
    end
    xlabel('Frames [1/25 s]');
    ylabel('[Nm]');
    
    linkaxes(sp,'x');
hold off

% plot identification only
xData = [1:51200] * 0.002441406;
figure()
grid on;
hold on
    title('Brake Torque Wheel Adapter');
    plot(xData(1:5120), t(1:5120),'r');
    plot(xData(1:5120), y(1:5120),'g');
%     plot(xData, t,'r');
%     plot(xData, y,'g');
    legend('Brake Torque Wheel Adapter', 'ANN');
    xlabel('Time [s]');
    ylabel('Brake Torque [Nm]');
    axis([0, 100, -2100, 800]);
    xlim([0,7])
hold off

% plot validation data in own figure
if plotValidateData
    figure()
    grid on;
    hold on
        title('Brake Torque Wheel Adapter Validation');
        plot(xData(1:5120), tVal(1:5120),'c');
        plot(xData(1:5120), yVal(1:5120),'k');
        legend('Brake Torque Validation', 'ANN Validation');
        xlabel('Time [s]');
        ylabel('Brake Torque [Nm]');
        axis([0, 100, -2400, 1100]);
        xlim([0,7])
%         ylim([-3000,2000])
    hold off
end