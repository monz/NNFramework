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
%     values = separateTrainingValues(input, target, 0.20, 0.05);

    % ------------------
    % fminunc
    % ------------------
%     costFcn = net.makeCostFcn(@nnfw.Util.mse, input, target);
% 
%     options = optimoptions('fminunc','GradObj','on', 'PlotFcns', {@optimplotfval, @optimplotstepsize}, 'MaxFunEvals', 30);
%     [x,y,exitFlag,output,g] = fminunc(costFcn,net.getWeightVector(),options);
%     g = g';

    % ------------------
    % lsqnonlin
    % ------------------
    g = 0; % TODO replace with gradient
%     costFcn = net.makeCostFcn2(@nnfw.Util.mse, input, target);
    costFcn = net.makeCostFcn2(@nnfw.Util.componentError, in, tn);
    
%     options = optimoptions('lsqnonlin', 'PlotFcns', {@optimplotfval, @optimplotstepsize, @optimplotx, @optimplotfirstorderopt});
%     options = optimoptions('lsqnonlin', 'Jacobian','on', 'PlotFcns', {@optimplotfval, @optimplotstepsize, @optimplotx, @optimplotfirstorderopt}, 'MaxIter', 30);
%     options = optimoptions('lsqnonlin', 'JacobMult','on','PlotFcns', {@optimplotfval, @optimplotstepsize, @optimplotx, @optimplotfirstorderopt});
    options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Jacobian','on','PlotFcns', {@optimplotfval, @optimplotstepsize, @optimplotx, @optimplotfirstorderopt});
%     options = optimoptions('lsqnonlin', 'Algorithm', 'trust-region-reflective', 'Jacobian','on','PlotFcns', {@optimplotfval, @optimplotstepsize, @optimplotx, @optimplotfirstorderopt});
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