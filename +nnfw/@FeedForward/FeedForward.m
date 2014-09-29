classdef FeedForward < nnfw.Network
    %FEEDFORWARD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function y = simulate(net, input)
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
            
%             transf1 = net.layers{1}.f;
%             a1 = transf1( net.IW{1}*input + net.b{1,1} );
%             transf2 = net.outputs{1}.f;
%             out = transf2( net.LW{2,1}*a1 + net.b{2,1} );
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

