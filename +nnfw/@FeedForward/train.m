function [E, g, output, lambda, jacobian] = train(net, input, target)
    % configure network layer sizes
    configure(net, input, target);
    % scale input/target values to prevent satturation of activation 
    % function in layer one
    in = nnfw.Util.minmaxMappingApply(input, net.minmaxInputSettings);
    tn = nnfw.Util.minmaxMappingApply(target, net.minmaxTargetSettings);
    
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
    [x, ~, ~, ~, output, lambda, jacobian] = lsqnonlin(costFcn,net.getWeightVector(), [], [], options);
    
    % set network weights found by optimization function
    net.setWeights(x);
    
    % calculate error and gradient
    gradient = net.makeCostFcn(@nnfw.Util.mseFast, in, tn);
    [E, g] = gradient(x);
end