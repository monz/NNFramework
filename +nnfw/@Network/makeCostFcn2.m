function costFcn = makeCostFcn2(net, fcn, input, target)

    costFcn = @CostFcn2;

    function [F, J] = CostFcn2(w)
        % set new network weights
        net.setWeights(w);

        % forward propagate with current weights
        % neuron outputs needed for backpropagation will be stored in a
        [y, a] = simulate(net, input, false);

        % calculate cost function
        Q = size(input,2); % number of training samples
        s_MSize = net.outputs{net.numLayers}.size;
        s_m = cell(Q, net.numLayers-1);
        if nargout > 1   % Two output arguments
            J = zeros(Q*s_MSize, net.getNumWeights());
        end
        % cost function
        F = bsxfun(fcn,y,target); % for performance improvement
        % load often used variables only once for performance improvements
        netSize = net.numLayers; % for performance improvement
        outputBpFcn = net.outputs{netSize}.f.backprop; % for performance improvement
        % calculate all marquardt sensitivities of last layer at once
        s_M = -outputBpFcn(y); % for performance improvement
        if nargout > 1   % Two output arguments
            for q = 1:Q
                % calculate remaining marquardt sensitivities
                % backward M-1, ..., 2, 1
                for layer = netSize-1:-1:1
                    bpFunction = net.layers{layer}.f.backprop;

                    % create derivated values matrix F_m
                    % diag creates a matrix with the values on the diagonal
                    % all other elements remain zero
                    F_m = diag(bpFunction(a{q, layer}));
                    % sensitivities
                    if ( layer == netSize-1 )
                        s_m{q, layer} = F_m * net.LW{layer+1, layer}' * s_M(q);
                    else
                        s_m{q, layer} = F_m * net.LW{layer+1, layer}' * s_m{q, layer+1};
                    end
                end
                
                for outputNr = 1:s_MSize % for every output index, calculate a row in the jacobian matrix
                    % generate jacobian matrix
                    offset = 0;
                    for layer = 1:netSize
                        if ( layer == 1 )
                            sensitivities = s_m{q, layer}(:, outputNr);
                            jEntriesWeights = sensitivities * input(:, q)';
                        elseif ( layer == netSize )
                            sensitivities = zeros(s_MSize, 1);
                            sensitivities(outputNr) = s_M(outputNr,q);
                            jEntriesWeights = sensitivities * a{q, layer-1}';
                        else
                            sensitivities = s_m{q, layer}(:, outputNr);
                            jEntriesWeights = sensitivities * a{q, layer-1}';
                        end
                        % prepare jacobian entries to be saved in a row of
                        % jacobian matrix
                        jEntriesWeights = reshape(jEntriesWeights', 1, numel(jEntriesWeights));
                        sensitivities = sensitivities(:)'; % error derived at bias
                        % save jacobian entries to the q-th jacobian matrix row
                        % jEntries of weights
                        startDim = offset+1;
                        offset = offset+length(jEntriesWeights);
                        
                        rowIdx = (q-1)*s_MSize + (outputNr);
                        J(rowIdx, startDim:offset) = J(rowIdx, startDim:offset) + jEntriesWeights;
                        
                        % jEntries of biases
                        startDim = offset+1;
                        offset = offset + length(sensitivities);
                        
                        J(rowIdx, startDim:offset) = J(rowIdx, startDim:offset) + sensitivities;
                    end 
                end
            end
        end
    end
end