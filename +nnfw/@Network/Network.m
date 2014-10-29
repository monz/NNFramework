classdef (Abstract) Network < handle
    %NET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numInputs = 0 % total number of network input vectors
        numLayers = 0 % number of network hidden layers
        numOutputs = 0 % total number of network output vectors
        
        inputs = {} % input layer information
        layers = {} % hidden layer information
        outputs = {} % output layer information
        
        biasConnect = []
        inputConnect = []
        layerConnect = []
        outputConnect = []
        
        IW = {} % input weights for each input vector
        LW = {} % layer weights for each hidden layer
        b = [] % biases
        
    end
    
    methods
        function initNetwork(obj, numInputs, numLayers, numOutputs)
            if numInputs <= 0 || numLayers <= 0 || numOutputs <= 0
                error('dimension of size zero not allowed');
            end
            
            obj.numInputs = numInputs;
            obj.numLayers = numLayers;
            obj.numOutputs = numOutputs;
            
            % -------------------------------------
            % default layer architecture and transfer functions
            % -------------------------------------
            % input layer
            for k = 1:numInputs
                obj.inputs{1,k} = nnfw.Layer('input', nnfw.Util.Activation.TANH);
            end
            % hidden layer
            for k = 1:numLayers-1
                obj.layers{1,k} = nnfw.Layer('hidden', nnfw.Util.Activation.TANH);
                obj.layers{1,k}.size = 10;
            end
            % output layer
%             for k = 1:numOutputs
            for k = 0:numOutputs-1
                obj.outputs{1,numLayers+k} = nnfw.Layer('output', nnfw.Util.Activation.PURELIN);
            end
        end
        
        function costFcn = makeCostFcn(net, fcn, input, target)

            costFcn = @CostFcn;
           
            function [E, gradients] = CostFcn(w)
                % set new network weights
                net.setWeights(w);
                
                % forward propagate with current weights
                % neuron outputs needed for backpropagation will be stored in a
                [y, a] = simulate(net, input);

                % calculate cost function
                Q = length(input); % number of training samples
                E = 0;
                s_M = zeros(1, Q);
                gradients = zeros(1, net.getNumWeights());
                for q = 1:Q
                    % cost function
                    E = E + fcn(y(q), target(:, q));

                    % calculate sensitivity of last layer
                    bpFunction = net.outputs{net.numLayers}.f.backprop;
                    s_M(q) = -2 * bpFunction(a{q, net.numLayers}) * (target(:, q) - y(q));

                    % calculate remaining sensitivities
                    % backward M-1, ..., 2, 1
                    s_m = cell(Q, net.numLayers-1);
                    for layer = net.numLayers-1:-1:1
                        bpFunction = net.layers{layer}.f.backprop;

                        % create derivated values matrix F_m
                        F_m = cell(Q, net.layers{layer}.size);
                        for j = 1:net.layers{layer}.size
                            % diag creates a matrix with the values on the diagonal
                            % all other elements remain zero
                            F_m{q, j} = diag(bpFunction(a{q, j}));
                        end
                        % sensitivities
                        if ( layer == net.numLayers-1 )
                            s_m{q, layer} = F_m{q, layer} * net.LW{layer+1}' * s_M(q);
                        else
                            s_m{q, layer} = F_m{q, layer} * net.LW{layer+1}' * s_m{q, layer+1};
                        end
                    end

                    % calculate gradients
                    offset = 0;
                    for layer = 1:net.numLayers 
                        if ( layer == 1 )
                            grads = s_m{q, layer} * input(:, q)';
                            bgrads = s_m{q, layer};
                        elseif (layer == net.numLayers)
                            grads = s_M(q) * a{q, layer-1}';
                            bgrads = s_M(q);
                        else
                            grads = s_m{q, layer} * a{q, layer-1}';
                            bgrads = s_m{q, layer};
                        end
                        % prepare gradients to be saved in a vector
                        grads = grads(:)';
                        bgrads = bgrads(:)';
                        % save gradients to the comprehensive gradient vector g
                        % gradients of weights
                        startDim = offset+1;
                        endDim = offset+length(grads);
                        gradients(1,startDim:endDim) = gradients(1,startDim:endDim) + grads;                
                        offset = endDim;
                        % gradients of biases
                        startDim = offset+1;
                        endDim = offset + length(bgrads);
                        gradients(1,startDim:endDim) = gradients(1,startDim:endDim) + bgrads;                
                        offset = endDim;                
                    end
                end
            end
        end
    end
    
    methods (Abstract)
        configure(obj, varargin);
        simulate(obj, input);
        train(obj, input, target);
        getNumWeights(obj);
        getGradientByWeight(obj, gradVector, weight);
        getWeightVector(obj);
        setWeights(obj);
    end
    
end

