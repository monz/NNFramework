function [E, g, output, jacobian] = train(net, input, target)
%TRAIN Adapts the neural network's weight values, supported by a optimization function.
%   After fully initializing the network, the training process is started.
%   A cost function is passed to a optimization function which then
%   searches a minimum while adapting the network's weight values
%   continuously.
%
%   [E, g, output, jacobian] = TRAIN(net, input, target)
%
%   net:        the neural network to be trained
%   input:      net input values
%   target:     target data the neural network should "learn"
%
%   Returns
%   E:          neural network's performance value, result of MSE evaluated
%               with the current weight values
%   g:          gradient vector evaluated with the current weight values
%   output:     results of the optimization function
%   jacobian:   jacobian matrix evaluated with the current weight values

    % configure network layer sizes
    configure(net, input, target);
    % scale input/target values to prevent satturation of activation 
    % function in layer one
    if net.optim.minmaxMapping
        in = nnfw.Util.minmaxMappingApply(input, net.minmaxInputSettings);
        tn = nnfw.Util.minmaxMappingApply(target, net.minmaxTargetSettings);
    else
        in = input;
        tn = target;
    end

    % initialize layer/bias-weights
    net.initWeights();
    
    % ------------------
    % separate input data into train, validate, test data
    % ------------------
    [values, net.valueIndexes] = nnfw.Util.separateTrainingValues(in, tn, net.optim.vlFactor, net.optim.tsFactor);
    in = values{1,1};
    tn = values{1,2};
    
    % ------------------
    % fminunc
    % ------------------
%     costFcn = net.makeCostFcn(@nnfw.Util.mse, in, tn);
% 
%     options = optimoptions('fminunc','GradObj','on', 'PlotFcns', {@optimplotfval, @optimplotstepsize}, 'MaxFunEvals', 30);
%     [x,y,exitFlag,output,g] = fminunc(costFcn,net.getWeightVector(),options);
%     g = g';
%     lambda = 0;
%     jacobian = 0;

    % ------------------
    % lsqnonlin
    % ------------------
    costFcn = net.makeCostFcn2(@nnfw.Util.componentError, in, tn);
    % register abort function
    abort = nnfw.Util.makeAbortFcn(net, values);
    % start cost function optimazation
    options = optimoptions('lsqnonlin', 'OutputFcn', abort, 'Algorithm', 'levenberg-marquardt', 'Jacobian','on','PlotFcns', net.optim.plotFcns, 'MaxIter', net.optim.maxIter);
    [x, ~, ~, ~, output, ~, jacobian] = lsqnonlin(costFcn,net.getWeightVector(), [], [], options);
    
    % set network weights found by optimization function
    net.setWeights(x);
    
    % calculate error and gradient
    gradient = net.makeCostFcn(@nnfw.Util.mseFast, in, tn);
    [E, g] = gradient(x);
end