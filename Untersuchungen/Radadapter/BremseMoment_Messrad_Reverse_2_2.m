%%% -----------------------------------------------------------------------
% =========================================================================
%   Wheel behavior Study Script
%   Train Method: SISO with Feedforward/Delay-Net
%
%   Activation for Bremse Moment on Measuring Wheel in Reverse Order
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
numNeurons = 10;
maxIter = 50;
delay1 = 1:20;
useToolbox = true;
delayNet = false;
flipTime = false;

% plot settings
figureNr = 2;
plotValidateData = true;

% load data settings

trainFile = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';
validateFile = '03_APRBS_APK1_sim3_DRV_100Pct_RSP.tim';

trdata = loadDataF(trainFile);
valdata = loadDataF(validateFile);

%% define input and target data

p = [trdata.MYMR'; trdata.FXMR'; trdata.FZMR'];
pVal = [valdata.MYMR'; valdata.FXMR'; valdata.FZMR'];

t = [trdata.Longitudinal_DeltaP'; trdata.Bremse_DeltaP'; trdata.Bremse_LVDT'];
tVal = [valdata.Longitudinal_DeltaP'; valdata.Bremse_DeltaP'; valdata.Bremse_LVDT'];


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
%     sp(1) = subplot(511);
%     plot(p(1,:),'r')
%     legend('Brake Torque');
%     
%     sp(2) = subplot(512);
%     plot(p(2,:),'r')
%     legend('Vertical Force');
    
    sp(3) = subplot(311);
    plot(t(1,:),'r')
    hold on
    plot(y(1,:),'g')
    if plotValidateData
        hold on
        plot(tVal(1,:),'c');
        hold on
        plot(yVal(1,:),'k');
        legend('Longitudinal DeltaP','ANN','Target Validation', 'ANN Validation')
    else
        legend('Longitudinal DeltaP','ANN')
    end
    xlabel('Time [s]');
    ylabel('[bar]');
    
    sp(4) = subplot(312);
    plot(t(2,:),'r')
    hold on
    plot(y(2,:),'g')
    if plotValidateData
        hold on
        plot(tVal(2,:),'c');
        hold on
        plot(yVal(2,:),'k');
        legend('Brake DeltaP','ANN','Target Validation', 'ANN Validation')
    else
        legend('Brake DeltaP','ANN');
    end
    xlabel('Time [s]');
    ylabel('[bar]');
    
    sp(5) = subplot(313);
    plot(t(3,:),'r')
    hold on
    plot(y(3,:),'g')    
    if plotValidateData
        hold on
        plot(tVal(3,:),'c');
        hold on
        plot(yVal(3,:),'k');
        legend('Brake LVDT','ANN','Target Validation', 'ANN Validation');
    else
        legend('Brake LVDT','ANN');
    end
    xlabel('Time [s]');
    ylabel('[mm]');

    linkaxes(sp,'xy');
hold off

% plot results in own figure
xData = [1:5120] * 0.002441406;

figure()
hold on
    grid on
    title('Wheel Adapter Values to Longitudinal DeltaP');
    plot(xData, t(1,1:5120),'r')
    plot(xData, y(1,1:5120),'g')
    legend('Longitudinal DeltaP','ANN')
    xlabel('Time [s]');
    ylabel('[bar]');
    axis([0,12,-45,35]);
hold off

figure()
hold on
    grid on
    title('Wheel Adapter Values to Brake DeltaP')
    plot(xData, t(2,1:5120),'r')
    plot(xData, y(2,1:5120),'g')
    legend('Brake DeltaP','ANN');
    xlabel('Time [s]');
    ylabel('[bar]');
    axis([0,12,-45,35]);
hold off

figure()
hold on
    grid on
    title('Wheel Adapter Values to Brake LVDT')
    plot(xData, t(3,1:5120),'r')
    plot(xData, y(3,1:5120),'g')    
    legend('Brake LVDT','ANN');
    xlabel('Time [s]');
    ylabel('[mm]');
    axis([0,12,-20,12]);
hold off

% plot validation data in own figure

 
figure()
hold on
    grid on
    title('Wheel Adapter Values to Longitudinal DeltaP Validation')
    plot(xData, tVal(1,1:5120),'c');
    plot(xData, yVal(1,1:5120),'k');
    legend('Longitudinal DeltaP Validation', 'ANN Validation')
    xlabel('Time [s]');
    ylabel('[bar]');
    axis([0,12,-45,35]);
hold off

figure()
hold on
    grid on
    title('Wheel Adapter Values to Brake DeltaP Validation')
    plot(xData, tVal(2,1:5120),'c');
    plot(xData, yVal(2,1:5120),'k');
    legend('Brake DeltaP Validation', 'ANN Validation')
    xlabel('Time [s]');
    ylabel('[bar]');
    axis([0,12,-45,35]);
hold off
       
figure()
hold on
    grid on
    title('Wheel Adapter Values to Brake LVDT Validation')
    plot(xData, tVal(3,1:5120),'c');
    plot(xData, yVal(3,1:5120),'k');
    legend('Brake LVDT Validation', 'ANN Validation');
    xlabel('Time [s]');
    ylabel('[mm]');
    axis([0,12,-20,12]);
hold off

if plotValidateData
    figure()
    hold on
        sp(1) = subplot(311);
        plot(tVal(1,:),'c');
        hold on
        plot(yVal(1,:),'k');
        legend('Longitudinal DeltaP Validation', 'ANN Validation')
        xlabel('Time [s]');
        ylabel('[bar]');

        sp(2) = subplot(312);
        plot(tVal(2,:),'c');
        hold on
        plot(yVal(2,:),'k');
        legend('Brake DeltaP Validation', 'ANN Validation')
        xlabel('Time [s]');
        ylabel('[bar]');

        sp(3) = subplot(313);
        plot(tVal(3,:),'c');
        hold on
        plot(yVal(3,:),'k');
        legend('Brake Validation', 'ANN Validation');
        xlabel('Time [s]');
        ylabel('[mm]');

        linkaxes(sp,'xy');
    hold off
end