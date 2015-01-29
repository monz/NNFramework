function gradient = getGradientByWeight(net, gradVector, weight)
%GETGRADIENTBYWEIGHT Returns the gradient of the cost function at the point of the given weight.
%
%   net:            a initialized and configured neural network
%   gradVector:     gradient vector containing all gradients
%   weight:         a vector describing the point w, whose gradient
%                   will be returned -> g(w). See example below.
%
%   Returns
%   gradient:       the gradient related to the given weight
%
%   weight is a vector of this form:
%   [layer, neuronDestination/biasNumber, neuronSource]
%   [int,   int,                          int,        ]
%
%   layer                           - specifies the layer in which the neuron/bias can be found
%   neuronDestination/biasNumber    - specifies the destination neuron / the number of the bias
%   neuronSource                    - specifies the source neuron / 0 for bias weights
%    
%   Example:
%   Gradient of w^1_(2,1)
%   [1, 2, 1]
%   layer - 1 for the first layer
%   neuronDestination - 2 because the weight goes to the 2nd neuron of layer one
%   neuronSource - 1 because it comes from the first input/neuron
%    
%   Example 2:
%   Gradien of b^2_1
%   [2, 1, 0]
%   layer - 2 for the second layer
%   biasNumber - 1 for the first bias in layer 2
%   neuronSource - 0 because its a bias weight

% calculate gradient position in gradients vector
    layer = weight(1);
    % move offset to layer-1
    offset = 0;
    for k = 1:layer-1
        if k == 1 
            offset = net.inputs{k}.size * net.layers{k}.size + net.layers{k}.size;
        elseif k < net.numLayers-1
            offset = offset + net.layers{k}.size * net.layers{k+1} + net.layers{k+1}.size;
        else
            offset = offset + net.layers{k-1}.size * net.outputs{k}.size + net.outputs{k}.size;
        end
    end

    % layer weight
    isNeuron = weight(3);
    if isNeuron
        neuronDestination = weight(2);
        neuronSource = weight(3);
        if layer == net.numLayers
            layersize = net.outputs{layer}.size;
        else
            layersize = net.layers{layer}.size;
        end
        offset = offset + (neuronSource-1) * layersize + neuronDestination;
    else
        % bias weight
        if layer == 1
            numNeurons = net.layers{layer}.size;
        else
            numNeurons = net.layers{layer-1}.size;
        end
        biasNumber = weight(2);
        offset = offset + numNeurons + biasNumber;
    end

    % get gradient
    gradient = gradVector(offset);
end