function initWeights( net )
%INITWEIGHTS Summary of this function goes here
%   Detailed explanation goes here
    offset = 0;
    weights = zeros(net.getNumWeights(), 1);
    for layer = 1:net.numLayers
        if layer == 1
            R = net.inputs{layer}.size;
            S_1 = net.layers{layer}.size;
            
            activationFcn = net.inputs{1, layer}.f;
            range = activationFcn.valueRange;
            numWeights = R*S_1 + S_1;            
        elseif layer == net.numLayers
            S_m = net.layers{layer-1}.size; % size of layer m-1
            S_M = net.outputs{layer}.size; % size of layer M
            
            activationFcn = net.outputs{layer}.f;
            range = activationFcn.valueRange;
            numWeights = S_m * S_M + S_M;
        else
            S_n = net.layers{layer-1}.size; % S_n = size of layer before current layer
            S_m = net.layers{layer}.size; % S_m = size of current layer
            
            activationFcn = net.layers{layer}.f;
            range = activationFcn.valueRange;
            numWeights = S_n * S_m + S_m;
        end
        startDim = offset + 1;
        endDim = offset + numWeights;
        offset = endDim;

        weights(startDim:endDim, 1) = weights(startDim:endDim,1) + nnfw.Util.getRangedRandomWeights(numWeights, range);
    end 
    net.setWeights(weights);
end

