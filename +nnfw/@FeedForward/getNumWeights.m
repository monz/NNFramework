function numWeights = getNumWeights(net)
%GETNUMWEIGHTS Returns the total number of weight values of the given neural network.
%   Calculates the total number of weight values related to the network
%   layer sizes, including the input and output layer dimensions.
%
%   net:          a initialized and configured neural network 
%
%   Returns
%   numWeights:   total number of weight values

    for layer = 1:net.numLayers
        if layer == 1 
            numWeights = net.inputs{layer}.size * net.layers{layer}.size + net.layers{layer}.size;
        elseif layer == net.numLayers
            numWeights = numWeights + net.layers{layer-1}.size * net.outputs{layer}.size + net.outputs{layer}.size;
        else
            numWeights = numWeights + net.layers{layer-1}.size * net.layers{layer}.size + net.layers{layer}.size;
        end
    end
    % TODO error if numWeights <= 0, indicates nn configuration error
end