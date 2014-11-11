function costFcn = makeCostFcn2(net, fcn, input, target)

    costFcn = @CostFcn2;

    function [F, J] = CostFcn2(w)
        % set new network weights
        net.setWeights(w);

        % forward propagate with current weights
        % neuron outputs needed for backpropagation will be stored in a
        [y, a] = simulate(net, input);
%         [y, ~] = simulate(net, input);

        % calculate cost function
        Q = length(input); % number of training samples
        F = zeros(Q,1);
        s_M = zeros(1, Q); % TODO convert to matrix, for multiple output neurons
        sMSize = net.outputs{net.numLayers}.size;
        s_m = cell(Q, net.numLayers-1); % TODO refactor dimensions, for multiple output neurons
        if nargout > 1   % Two output arguments
            J = zeros(Q*sMSize, net.getNumWeights());
        end
        for q = 1:Q
            % cost function
            F(q) = fcn(y(q), target(:, q));

            if nargout > 1   % Two output arguments
                % calculate marquardt sensitivity of last layer
                bpFunction = net.outputs{net.numLayers}.f.backprop;
                s_M(q) = -bpFunction(a{q, net.numLayers}); % TODO put into right matrix position for multiple output neurons

                % calculate remaining marquardt sensitivities
                % backward M-1, ..., 2, 1
    %             s_m = cell(Q, net.numLayers-1); % TODO refactor dimensions, for multiple output neurons
                for layer = net.numLayers-1:-1:1
                    bpFunction = net.layers{layer}.f.backprop;

                    % create derivated values matrix F_m
                    % diag creates a matrix with the values on the diagonal
                    % all other elements remain zero
                    F_m = diag(bpFunction(a{q, layer}));
                    % sensitivities
                    if ( layer == net.numLayers-1 )
                        s_m{q, layer} = F_m * net.LW{layer+1, layer}' * s_M(q);
                    else
                        s_m{q, layer} = F_m * net.LW{layer+1, layer}' * s_m{q, layer+1};
                    end
                end

                % generate jacobian matrix
                offset = 0;
                for layer = 1:net.numLayers 
                    if ( layer == 1 )
                        grads = s_m{q, layer} * input(:, q)';
                        bgrads = s_m{q, layer};
                    elseif ( layer == net.numLayers )
                        grads = s_M(q) * a{q, layer-1}';
                        bgrads = s_M(q);
                    else
                        grads = s_m{q, layer} * a{q, layer-1}';
                        bgrads = s_m{q, layer};
                    end
                    % prepare gradients to be saved in a vector
                    grads = grads(:)';
                    bgrads = bgrads(:)';
                    % save gradients to the comprehensive gradient vector g
                    % gradients of weights
                    startDim = offset+1;
                    endDim = offset+length(grads);
                    J(q,startDim:endDim) = J(q,startDim:endDim) + grads;                
                    offset = endDim;
                    % gradients of biases
                    startDim = offset+1;
                    endDim = offset + length(bgrads);
                    J(q,startDim:endDim) = J(q,startDim:endDim) + bgrads;                
                    offset = endDim;                
                end 
            end
        end
    end
end