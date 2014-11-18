function setWeights(net, weights)
    offset = 0;
    for layer = 1:net.numLayers
        if layer == 1
            % layer weights
            R = net.inputs{layer}.size;
            S_1 = net.layers{layer}.size;
            net.IW{layer} = zeros(S_1, R);
            for k = 1:S_1
                startDim = offset + 1;
                endDim = offset + R;
                % set weights in k-th row of input weights matrix
                % of layer-th layer
                net.IW{layer}(k,:) = weights(startDim:endDim,1);
                offset = endDim;
            end
            % bias weights
            startDim = offset +1;
            endDim = offset + net.layers{layer}.size; 
            net.b{layer} = weights(startDim:endDim,1);
            offset = endDim;
        elseif layer == net.numLayers
            % layer weights
            startDim = offset +1;
            S_m = net.layers{layer-1}.size; % size of layer m-1
            S_M = net.outputs{layer}.size; % size of layer M
            endDim = offset + S_m * S_M;
            net.LW{layer,layer-1} = weights(startDim:endDim,1)';
            offset = endDim;
            % bias weights
            startDim = offset +1;
            endDim = offset + S_M;
            net.b{layer} = weights(startDim:endDim,1);
            offset = endDim;                    
        else
            % layer weights
            S_n = net.layers{layer-1}.size; % S_n = size of layer before current layer
            S_m = net.layers{layer}.size; % S_m = size of current layer
            net.LW{layer,layer-1} = zeros(S_m, S_n);                    
            for k = 1:S_m
                startDim = offset + 1;
                endDim = offset + S_n;
                % set weights in k-th row of weights matrix
                % of layer-th layer
                net.LW{layer, layer-1}(k,:) = weights(startDim:endDim,1);
                offset = endDim;
            end
            % bias weights
            startDim = offset +1;
            endDim = offset + net.layers{layer}.size; 
            net.b{layer} = weights(startDim:endDim,1);
            offset = endDim;
        end
    end            
end