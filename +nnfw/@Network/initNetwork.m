function initNetwork(net, hiddenLayerSizes)
%INITNETWORK Initializes the neural network.
%   Each layer of the neural network gets initialized with its size and the
%   default activation function, which is the tanh function in the hidden
%   layers and the linear function in the outputlayer. Also the type of the
%   layers is set. This is for clearness only.
%
%   INITNETWORK(net, hiddenLayerSizes)
%
%   net:                the neural network which get initialized
%
%   hiddenLayerSizes:   vector of layer sizes, e.g. [10 2 10] describes
%                       three hidden layers with 10 neurons in the first,
%                       2 neurons in the second and 10 neurons in the third
%                       layer

    numLayers = length(hiddenLayerSizes) + 1; % plus one for output layer

    if ismember(0, hiddenLayerSizes)
        error('dimension of size zero not allowed');
    elseif numLayers <= 0
        error('no hidden layer sizes specified');
    end

    net.numLayers = numLayers;

    % -------------------------------------
    % default layer architecture and transfer functions
    % -------------------------------------
    for k = 1:numLayers
       if k == 1
          net.inputs{1,k} = nnfw.Layer('input', nnfw.Util.Activation.TANH); 

          net.layers{1,k} = nnfw.Layer('hidden', nnfw.Util.Activation.TANH);
          net.layers{1,k}.size = hiddenLayerSizes(k);
       elseif k <= numLayers-1
          net.layers{1,k} = nnfw.Layer('hidden', nnfw.Util.Activation.TANH);
          net.layers{1,k}.size = hiddenLayerSizes(k);
       else
          net.outputs{1,numLayers} = nnfw.Layer('output', nnfw.Util.Activation.PURELIN);
       end
    end
end