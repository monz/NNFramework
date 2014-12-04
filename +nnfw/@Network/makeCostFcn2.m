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
        s_M = zeros(size(target));
        s_MSize = net.outputs{net.numLayers}.size;
        s_m = cell(Q, net.numLayers-1);
        if nargout > 1   % Two output arguments
            J = zeros(Q*s_MSize, net.getNumWeights());
        end
        % cost function
        F = bsxfun(@minus,target,y); % for performance enhancement
        netSize = net.numLayers; % for performance enhancement
        for q = 1:Q
            if nargout > 1   % Two output arguments
                % calculate marquardt sensitivity of last layer
                bpFunction = net.outputs{netSize}.f.backprop;
                s_M(:, q) = -bpFunction(a{q, netSize});

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
                            jEntriesWeights = s_m{q, layer}(:, outputNr) * input(:, q)';
                            jEntriesBias = s_m{q, layer}(:, outputNr);
                            layerSize = net.inputs{layer}.size;
                            layerP1Size = net.layers{layer}.size;
                        elseif ( layer == netSize )
                            s_MVector = zeros(s_MSize, 1);
                            s_MVector(outputNr) = s_M(outputNr,q);
                            jEntriesWeights = s_MVector * a{q, layer-1}';
                            jEntriesBias = s_MVector;
                            layerSize = net.layers{layer-1}.size;
                            layerP1Size = net.outputs{layer}.size;
                        else
                            jEntriesWeights = s_m{q, layer}(:, outputNr) * a{q, layer-1}';
                            jEntriesBias = s_m{q, layer}(:, outputNr);
                            layerSize = net.layers{layer-1}.size;
                            layerP1Size = net.layers{layer}.size;
                        end
                        % prepare jacobian entries to be saved in a row of
                        % jacobian matrix
                        jEntriesWeights2 = zeros(1, layerSize*layerP1Size);
                        offset2 = 0;
                        for k = 1:layerP1Size
                            startDim2 = offset2 +1;
                            endDim2 = offset2 + layerSize;
                            jEntriesWeights2(1, startDim2:endDim2) = jEntriesWeights2(1, startDim2:endDim2) + jEntriesWeights(k,:); % error derived at weights
                            offset2 = endDim2;
                        end
                        jEntriesWeights = jEntriesWeights2;
                        jEntriesBias = jEntriesBias(:)'; % error derived at bias
                        % save jacobian entries to the q-th jacobian matrix row
                        % jEntries of weights
                        startDim = offset+1;
                        endDim = offset+length(jEntriesWeights);
                        
                        rowIdx = (q-1)*s_MSize + (outputNr);
                        J(rowIdx, startDim:endDim) = J(rowIdx, startDim:endDim) + jEntriesWeights;                
                        
                        offset = endDim;
                        % jEntries of biases
                        startDim = offset+1;
                        endDim = offset + length(jEntriesBias);
                        
                        J(rowIdx, startDim:endDim) = J(rowIdx, startDim:endDim) + jEntriesBias;                
                        offset = endDim;                
                    end 
                end
            end
        end
    end
end