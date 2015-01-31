function configure(net, input, target)
%CONFIGURE Sets the input and target layer sizes related to the input and target value dimension.
%   Additionally the minMaxMapping settings for the input and target values
%   get extracted and saved into the network. The minMaxMapping settings
%   are used while network training and simulation to provide better
%   performance.
%
%   CONFIGURE(net, input, target)
%
%   net:        the neural network to be configured
%   input:      input values, whose dimension gets extracted
%   target:     target values, whose dimension gets extracted

    % -------------------------------------
    % define network dimensions
    % -------------------------------------
    % extract min/max value information from input data
    [~,net.minmaxInputSettings] = nnfw.Util.minmaxMapping(input);
    % for further use convert input to cell array
    if ~iscell(input)
        input = {input};
    end

    % check given input size against network input size
    if size(input,1) ~= net.numInputs
        error('input parameter dimension missmatch');
    end
    % define input sizes
    for k = 1:net.numInputs
        net.inputs{k}.size = size(input{k,:},1);
    end

    % extract min/max value information from target data
    [~,net.minmaxTargetSettings] = nnfw.Util.minmaxMapping(target);
    % for further use convert output to cell array
    if ~iscell(target)
        target = {target};
    end

    % check given target size against network output size
    if size(target,1) ~= net.numOutputs
        error('output parameter dimension missmatch');
    end
    % define output sizes
    for k = 0:net.numOutputs-1
        net.outputs{net.numLayers+k}.size = size(target{k+1,:},1);
    end
end