classdef FeedForward < nnfw.Network
    %FEEDFORWARD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
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
                        LW = net.LW{layer,1};
                        p = a{q, layer-1};
                        transf = net.layers{layer}.f.f;
                    end
                    a{q, layer} = transf( LW*p + net.b{layer} );
                end

                y(q) = a{q,net.numLayers};
            end
        end
        
        function [E, g] = train(net, input, target)
            % configure network layer sizes
            configure(net, input, target);
            
            % forward propagate with current weights
            % neuron outputs needed for backpropagation will be stored in a
            [y, a] = simulate(net, input);
            
            % calculate cost function
            Q = length(input); % number of training samples
            E = zeros(1, Q);
            s_M = zeros(1, Q);
            g = zeros(1, net.getNumWeights());
            for q = 1:Q
                E(q) = nnfw.Util.mse(y(q), target(:, q));

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
                    g(1,startDim:endDim) = g(1,startDim:endDim) + grads;                
                    offset = endDim;
                    % gradients of biases
                    startDim = offset+1;
                    endDim = offset + length(bgrads);
                    g(1,startDim:endDim) = g(1,startDim:endDim) + bgrads;                
                    offset = endDim;                
                end
            end
        end
        
        function net = FeedForward(numInputs, numLayers, numOutputs)
            net.initNetwork(numInputs, numLayers, numOutputs);
            
            % -------------------------------------
            % define feedforward layer connections
            % -------------------------------------
            net.biasConnect = ones(1, numLayers);
            
            net.inputConnect = zeros(1, numLayers);
            net.inputConnect(1) = 1; % input connects to layer 1 (inputConnect(layer))
            
            net.outputConnect = zeros(1, numLayers);
            net.outputConnect(numLayers) = 1; % last layer is set to output layer
            
            net.layerConnect = zeros(numLayers);
            % connect each layer to its fellow layer (feedforward)
            for k = 2:numLayers
                net.layerConnect(k, k-1) = 1; % row = destination layer; column = source layer
            end
        end
        
        function configure(net, varargin)
            % expected parameter =  net, input, traget
            % input and target have to be cell arrays
            % x-th row in input/target cell array defines x-th input/output
            if nargin < 3
                error('not enough arguments');
            elseif nargin > 3
                error('to much arguments');
            end

            % -------------------------------------
            % define network dimensions
            % -------------------------------------
            input = varargin{1};
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
            
            % only set size while network initialization
            % to prevent overriding user settings
%             % set default hidden layer size
%             for k = 1:net.numLayers - 1
%                 net.layers{k}.size = 10;
%             end
            
            output = varargin{2};
            if ~iscell(output)
                output = {output};
            end
            
            % check given target size against network output size
            if size(output,1) ~= net.numOutputs
                error('output parameter dimension missmatch');
            end
            % define output sizes
            for k = 0:net.numOutputs-1
                net.outputs{net.numLayers+k}.size = size(output{k+1,:},1);
            end
        end
        
        function numWeights = getNumWeights(net) 
            for k = 1:net.numLayers
                if k == 1 
                    numWeights = net.inputs{k}.size * net.layers{k}.size + net.layers{k}.size;
                elseif k < net.numLayers-1
                    numWeights = numWeights + net.layers{k}.size * net.layers{k+1} + net.layers{k+1}.size;
                else
                    numWeights = numWeights + net.layers{k-1}.size * net.outputs{k}.size + net.outputs{k}.size;
                end
            end
        end
        
        function gradient = getGradientByWeight(net, gradVector, weight)
            % weight is a vector of this form
            % [layer, neuronDestination/biasNumber, neuronSource, isNeuron]
            % [int,   int,                          int,          bool ]
            %
            % layer - specifies the layer in which the neuron/bias could be
            % found
            %
            % neuronDestination/biasNumber - specifies the destination
            % neuron / the number of the bias
            %
            % neuronSource - specifies the source neuron / 0 for bias
            % weights
            %
            % isNeuron - false if you want to address a bias weight
            %
            % Example:
            % Gradient of w^1_(2,1)
            % [1, 2, 1, 1]
            % layer - 1 for the first layer
            % neuronDestination - 2 because the weight goes to the 2nd
            % neuron of layer one
            % neuronSource - 1 because it comes from the first input/neuron
            % isNeuron - 1 (true) because its no bias weight
            %
            % Example 2:
            % Gradien of b^2_1
            % [2, 1, 0, 0]
            % layer - 2 for the second layer
            % biasNumber - 1 for the first bias in layer 2
            % neuronSource - 0 because its a bias weight
            % isNeuron - 0 (false) because its a bias weight
            
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
            isNeuron = weight(4);
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
    end
    
end

