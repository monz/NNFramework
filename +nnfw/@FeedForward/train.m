function [E, g, output, lambda, jacobian] = train(net, input, target)
    % configure network layer sizes
    configure(net, input, target);
    
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
    costFcn = net.makeCostFcn2(@nnfw.Util.mse, input, target);
    
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
    [y, ~] = simulate(net, input);

    % calculate cost function
    Q = length(input); % number of training samples
    E = 0;
    for q = 1:Q
        % cost function
        E = E + nnfw.Util.mse(y(q), target(:, q));
    end
end