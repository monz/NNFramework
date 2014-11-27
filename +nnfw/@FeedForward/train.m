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
    values = nnfw.Util.separateTrainingValues(in, tn, 0.20, 0.05);
    in = values{1,1};
    tn = values{1,2};
%     trainValues = values{1,1};
%     trainTargets = values{1,2};
    testValues = values{2,1};
    testTargets = values{2,2};
    validateValues = values{3,1};
    validateTargets = values{3,2};
    
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
    g = 0; % TODO replace with gradient
    costFcn = net.makeCostFcn2(@nnfw.Util.componentError, in, tn);
    
%     options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Jacobian','on','PlotFcns', {@optimplotfval, @optimplotstepsize, @optimplotx, @optimplotfirstorderopt});
    abort = nnfw.Util.makeAbortFcn(net, validateValues, validateTargets);
    options = optimoptions('lsqnonlin', 'OutputFcn', abort, 'Algorithm', 'levenberg-marquardt', 'Jacobian','on','PlotFcns', {@optimplotfval, @optimplotstepsize, @optimplotx, @optimplotfirstorderopt});
    [x, ~, ~, ~, output, lambda, jacobian] = lsqnonlin(costFcn,net.getWeightVector(), [], [], options);
    
    % set network weights found by optimization function
    net.setWeights(x);

    % forward propagate with current weights
    % neuron outputs needed for backpropagation will be stored in a
    [y, ~] = simulate(net, in, false);

    % calculate cost function
    Q = length(in); % number of training samples
    E = 0;
    for q = 1:Q
        % cost function
        E = E + nnfw.Util.mse(y(:, q), tn(:, q));
    end
end