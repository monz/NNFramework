function weightVector = getWeightVector(net)
    weightVector = zeros(net.getNumWeights(),1);
    offset = 0;
    for layer = 1:net.numLayers
        if layer == 1
            % layer weights
            R = net.inputs{layer}.size;
            S_1 = net.layers{layer}.size;
            for k = 1:S_1
                startDim = offset +1;
                endDim = offset + R;
                weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.IW{layer}(k, :)';            
                offset = endDim;
            end
            % bias weights
            startDim = offset +1;
            endDim = offset + net.layers{layer}.size; 
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.b{layer};
            offset = endDim;
        elseif layer == net.numLayers
            % layer weights
            S_m = net.layers{layer-1}.size; % size of layer m-1
            S_M = net.outputs{layer}.size; % size of layer M
            for k = 1:S_M
                startDim = offset +1;
                endDim = offset + S_m;
                weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.LW{layer,layer-1}(k, :)';
                offset = endDim;
            end
            % bias weights
            startDim = offset +1;
            endDim = offset + net.outputs{layer}.size;
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.b{layer};
            offset = endDim;                    
        else
            % layer weights
            S_n = net.layers{layer-1}.size; % S_n = size of layer before current layer
            S_m = net.layers{layer}.size; % S_m = size of current layer
            for k = 1:S_m
                startDim = offset +1;                    
                endDim = offset + S_n;
                weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.LW{layer,layer-1}(k, :)';
                offset = endDim;
            end
            % bias weights
            startDim = offset +1;
            endDim = offset + net.layers{layer}.size; 
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.b{layer};
            offset = endDim;
        end
    end
end