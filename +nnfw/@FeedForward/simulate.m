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
    Q = length(input);
    a = cell(Q,net.numLayers);
    outputSize = net.outputs{net.numLayers}.size;
    y = zeros(outputSize,Q);
    for q = 1:Q
        for layer = 1:net.numLayers
            if layer == 1 % input layer
                LW = net.IW{layer};
                p = input(:,q);
                transf = net.layers{layer}.f.f;
            elseif layer == net.numLayers % output layer
                LW = net.LW{layer,layer-1};
                p = a{q, layer-1};
                transf = net.outputs{net.numLayers}.f.f;
            else % hidden layer
                LW = net.LW{layer,layer-1};
                p = a{q, layer-1};
                transf = net.layers{layer}.f.f;
            end
            a{q, layer} = transf( LW*p + net.b{layer} );
        end
        if applyValueMapping
            y(:,q) = nnfw.Util.minmaxMappingRevert(a{q,net.numLayers}, net.minmaxTargetSettings);
        else
            y(:,q) = a{q,net.numLayers};
        end
    end
end