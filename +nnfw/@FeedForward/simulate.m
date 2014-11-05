function [y, a] = simulate(net, input)
    % -------------------------------------
    % feed forward
    % -------------------------------------
    Q = length(input);
    a = cell(Q,net.numLayers);
    y = zeros(1, Q);
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

        y(q) = a{q,net.numLayers};
    end
end