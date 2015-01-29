function [y, a] = simulate(net, varargin)
%SIMULATE Adapts the neural network's weight values, supported by a optimization function.
%   After fully initializing the network, the training process is started.
%   A cost function is passed to a optimization function which then
%   searches a minimum while adapting the network's weight values
%   continuously.
%
%   [y,a] = simulate(net, p)
%   [y,a] = simulate(net, p, applyValueMapping)
%
%   net:                the neural network which is used to simulate the
%                       input values
%   p:                  net input values
%   applyValueMapping:  if true the input values get mapped to the range of
%                       [-1:1].
%                       Attention: Value mapping is applied row based!
%
%   Returns
%   y:                  the network's response to the input values
%   a:                  each layer's neurons' outputs

    if nargin == 2
        applyValueMapping = net.optim.minmaxMapping;
    elseif nargin == 3
        applyValueMapping = varargin{2};
    else
        error('invalid argument count');
    end
    
    % scale input/target values to prevent satturation of activation 
    % function in layer one
    if applyValueMapping
        input = nnfw.Util.minmaxMappingApply(varargin{1}, net.minmaxInputSettings);
    else
        input = varargin{1};
    end
    
    % -------------------------------------
    % feed forward
    % -------------------------------------
    Q = size(input,2);
    a = cell(Q,net.numLayers-1);
    % load often used variables only once for performance improvements
    netSize = net.numLayers;
    inputTransFcn = net.layers{1}.f.f;
    inputLW = net.IW{1};
    inputBias = net.b{1};
    outputTransFcn = net.outputs{netSize}.f.f;
    outputLW = net.LW{netSize,netSize-1};
    outputBias = net.b{netSize};

    % currentOut varialbe describes the current layers computed output
    % currentOut is always given to the next layer as input
    % compute input layer completely
    currentOut = inputTransFcn(bsxfun(@plus,inputLW*input,inputBias));
    a(:,1) = nnfw.Util.myNum2Cell(currentOut);
    % compute each hidden layer completely
    for layer = 2:netSize-1
        LW = net.LW{layer,layer-1};
        currentOut = net.layers{layer}.f.f(bsxfun(@plus,LW*currentOut,net.b{layer}));
        a(:,layer) = nnfw.Util.myNum2Cell(currentOut);
    end
    % compute output layer completely
    y = outputTransFcn(bsxfun(@plus,outputLW*currentOut,outputBias));
    if applyValueMapping
        y = nnfw.Util.minmaxMappingRevert(y, net.minmaxTargetSettings);
    end
    % --------------------------------------
    % map outputs to 0...1 if is pattern net
    % --------------------------------------
    if net.isPatternNet && applyValueMapping
        y_d = zeros(size(y));
        % extract indexes of max value of each column
        [~, i] = max(y, [], 1);
        % map max values to given value
        y_d(sub2ind(size(y), i, 1:length(i))) = 1;
        y = y_d;
    end
end