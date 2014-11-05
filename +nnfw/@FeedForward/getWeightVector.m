function weightVector = getWeightVector(net)
    weightVector = zeros(net.getNumWeights(),1);
    offset = 0;
    for layer = 1:net.numLayers
        if layer == 1
            % layer weights
            startDim = offset +1;
            endDim = net.inputs{layer}.size * net.layers{layer}.size;
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.IW{layer}(:);
            offset = endDim;
            % bias weights
            startDim = offset +1;
            endDim = offset + net.layers{layer}.size; 
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.b{layer};
            offset = endDim;
        elseif layer == net.numLayers
            % layer weights
            startDim = offset +1;
            endDim = offset + net.layers{layer-1}.size;
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.LW{layer,layer-1}(:);
            offset = endDim;
            % bias weights
            startDim = offset +1;
            endDim = offset + net.outputs{layer}.size;
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.b{layer};
            offset = endDim;                    
        else
            % layer weights
            startDim = offset +1;                    
            endDim = offset + net.layers{layer-1}.size * net.layers{layer}.size;
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.LW{layer,layer-1}(:);
            offset = endDim;
            % bias weights
            startDim = offset +1;
            endDim = offset + net.layers{layer}.size; 
            weightVector(startDim:endDim,1) = weightVector(startDim:endDim,1) + net.b{layer};
            offset = endDim;
        end
    end
end