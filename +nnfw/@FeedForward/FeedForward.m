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
            a = cell(1,net.numLayers);
            for layer = 1:net.numLayers
                if layer == 1 % input layer
                    LW = net.IW{layer};
                    p = input;
                    transf = net.layers{layer}.f.f;
                    a{layer} = transf( LW*p + net.b{layer} );
                elseif layer == net.numLayers % output layer
                    LW = net.LW{layer,layer-1};
                    p = a{layer-1};
%                     transf = net.outputs{net.numLayers - 1}.f;
                    transf = net.outputs{1}.f.f;
                else % hidden layer
                    LW = net.LW{layer,1};
                    p = a{layer-1};
                    transf = net.layers{layer}.f.f;
                end
                a{layer} = transf( LW*p + net.b{layer} );
            end
            
            y = a{1,net.numLayers};
        end
        
        function [e, g, bGrad] = train(net, input, target)
            % forward propagate with current weights
            % neuron outputs needed for backpropagation will be stored in a
            [y, a] = simulate(net, input);
            
            % calculate cost function
            e = nnfw.Util.mse(y, target);
            
            % calculate sensitivity of last layer
            s_M = -2 * 1 * (target - y); % ... * 1 * ... because linear derivated = 1
            
            % calculate remaining sensitivities
            % backward M-1, ..., 2, 1
            s_m = cell(1, net.numLayers-1);
            for k = net.numLayers-1:-1:1
                bpFunction = net.layers{k}.f.backprop;
                
                % create derivated values matrix F_m
                F_m = cell(1, net.layers{k}.size);
                for j = 1:net.layers{k}.size
                    % diag creates a matrix with the values on the diagonal
                    % all other elements remain zero
                    F_m{j} = diag(bpFunction(a{k, j}));
                end
                % sensitivities
                if ( k == net.numLayers-1 )
                    s_m{k} = F_m{k} * net.LW{k+1}' * s_M;
                else
                    s_m{k} = F_m{k} * net.LW{k+1}' * s_m{k+1};
                end
            end
            
            % calculate gradients
            g = cell(1, net.numLayers);
            for k = 1:net.numLayers 
                if ( k == 1 )
                    g{k} = s_m{k} * input';
                elseif (k == net.numLayers)
                    g{k} = s_M * a{k-1}';
                else
                    g{k} = s_m{k} * a{k-1}';
                end
            end
            
            % bias gradients
            bGrad = cell(1,net.numLayers);
            for k = 1:net.numLayers 
                if ( k == net.numLayers )
                    bGrad{k} = s_M;
                else
                    bGrad{k} = s_m{k};
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
            % check given input size against network input size
            if size(input,1) ~= net.numInputs
                error('input parameter dimension missmatch');
            end
            % define input sizes
            for k = 1:net.numInputs
                net.inputs{k}.size = size(input{k,:},1);
            end
            
            % set default hidden layer size
            for k = 1:net.numLayers - 1
                net.layers{k}.size = 10;
            end
            
            output = varargin{2};
            % check given target size against network output size
            if size(output,1) ~= net.numOutputs
                error('output parameter dimension missmatch');
            end
            % define output sizes
            for k = 1:net.numOutputs
                net.outputs{k}.size = size(output{k,:},1);
            end
        end
    end
    
end

