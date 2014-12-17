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
            % calculate remaining marquardt sensitivities
            % backward M-1, ..., 2, 1
            for layer = netSize-1:-1:1
                bpFunction = net.layers{layer}.f.backprop;
                LWtransp = net.LW{layer+1, layer}';
                % calculcate the derivated values of the layer neuron's
                % outputs for all training values at once - these values
                % are prepared for the F_m diagonal-matrix
                dValues = bpFunction(reshape([a{:,layer}], size(a{1,layer},1), size(a,1))');
                if ( layer == netSize-1 )
                    for q = 1:Q
                        % create derivated values matrix F_m
                        % diag creates a matrix with the values on the diagonal
                        % all other elements remain zero
                        F_m = diag(dValues(q,1:end));
                        % sensitivities
                        % TODO check if computation is correct here, maybe each
                        % outputNr need separate computation see Script Prof.Endisch
                        % chapter 5 jacobian computation conclusion senction 2
                        % for linear output function it is correct - all
                        % s_M values are equal to -1
                        s_m{q, layer} = F_m * LWtransp * s_M(q);
                    end
                else
                    for q = 1:Q
                        % create derivated values matrix F_m
                        % diag creates a matrix with the values on the diagonal
                        % all other elements remain zero
                        F_m = diag(dValues(q,1:end));
                        % sensitivities
                        s_m{q, layer} = F_m * LWtransp * s_m{q, layer+1};
                    end
                end
            end

            % generate jacobian matrix
            for q = 1:Q
                for outputNr = 1:s_MSize % for every output index, calculate a row in the jacobian matrix
                    offset = 0;
                    rowIdx = (q-1)*s_MSize + (outputNr);
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
                        % save jacobian entries to the q-th jacobian matrix row
                        % jEntries of weights
                        startDim = offset+1;
                        offset = offset+length(jEntriesWeights);
                        
                        J(rowIdx, startDim:offset) = jEntriesWeights;
                        
                        % jEntries of biases
                        startDim = offset+1;
                        offset = offset + length(sensitivities);
                        
                        J(rowIdx, startDim:offset) = sensitivities;
                    end 
                end
            end
        end
    end
end