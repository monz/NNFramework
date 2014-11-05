function numWeights = getNumWeights(net) 
    for layer = 1:net.numLayers
        if layer == 1 
            numWeights = net.inputs{layer}.size * net.layers{layer}.size + net.layers{layer}.size;
        elseif layer == net.numLayers
            numWeights = numWeights + net.layers{layer-1}.size * net.outputs{layer}.size + net.outputs{layer}.size;
        else
            numWeights = numWeights + net.layers{layer-1}.size * net.layers{layer}.size + net.layers{layer}.size;
        end
    end
    % TODO error if numWeights <= 0, indicates nn configuration
    % error
end