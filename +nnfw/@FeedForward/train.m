function [E, g] = train(net, input, target)
    % configure network layer sizes
    configure(net, input, target);

    costFcn = net.makeCostFcn(@nnfw.Util.mse, input, target);

    options = optimoptions('fminunc','GradObj','on', 'PlotFcns', {@optimplotfval, @optimplotstepsize}, 'MaxFunEvals', 30);
    [x,y,exitFlag,output,g] = fminunc(costFcn,net.getWeightVector(),options);
    g = g';
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