function [y, a] = simulate(net, varargin)

    if nargin == 2
        applyValueMapping = true;
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
    a = cell(Q,net.numLayers);
    outputSize = net.outputs{net.numLayers}.size;
    % load often used variables only once for performance improvements
    netSize = net.numLayers; % for performance improvement
    inputTransFcn = net.layers{1}.f.f; % for performance improvement
    inputLW = net.IW{1}; % for performance improvement
    outputTransFcn = net.outputs{netSize}.f.f; % for performance improvement
    outputLW = net.LW{netSize,netSize-1}; % for performance improvement
    y = zeros(outputSize,Q);
    for q = 1:Q
        for layer = 1:netSize
            if layer == 1 % input layer
                p = input(:,q);
                a{q, layer} = inputTransFcn( inputLW*p + net.b{layer} );
            elseif layer == netSize % output layer
                p = a{q, layer-1};
                a{q, layer} = outputTransFcn( outputLW*p + net.b{layer} );
            else % hidden layer
                LW = net.LW{layer,layer-1};
                p = a{q, layer-1};
                a{q, layer} = net.layers{layer}.f.f( LW*p + net.b{layer} );
            end
        end
        if applyValueMapping
            y(:,q) = nnfw.Util.minmaxMappingRevert(a{q,netSize}, net.minmaxTargetSettings);
        else
            y(:,q) = a{q,netSize};
        end
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