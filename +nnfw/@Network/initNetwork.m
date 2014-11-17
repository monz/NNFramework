function initNetwork(obj, hiddenLayerSizes)
    numLayers = length(hiddenLayerSizes) + 1; % plus one for output layer

    if ismember(0, hiddenLayerSizes)
        error('dimension of size zero not allowed');
    elseif numLayers <= 0
        error('no hidden layer sizes specified');
    end

    obj.numLayers = numLayers;

    % -------------------------------------
    % default layer architecture and transfer functions
    % -------------------------------------
    for k = 1:numLayers
       if k == 1
          obj.inputs{1,k} = nnfw.Layer('input', nnfw.Util.Activation.TANH); 

          obj.layers{1,k} = nnfw.Layer('hidden', nnfw.Util.Activation.TANH);
          obj.layers{1,k}.size = hiddenLayerSizes(k);
       elseif k <= numLayers-1
          obj.layers{1,k} = nnfw.Layer('hidden', nnfw.Util.Activation.TANH);
          obj.layers{1,k}.size = hiddenLayerSizes(k);
       else
          obj.outputs{1,numLayers} = nnfw.Layer('output', nnfw.Util.Activation.PURELIN);
       end
    end
end