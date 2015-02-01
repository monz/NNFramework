classdef FeedForward < nnfw.Network
    %FEEDFORWARD Contains special attributes and methods for the feedforward neural network type.
    %   FEEDFORWARD is a subclass of the abstract network class. A
    %   object of this class inherits all attributes of the parent class.
    %   This type of neural network has only forward connections.
    %
    %   See also HANDLE, NETWORK.
    
    properties
    end
    
    methods
        function net = FeedForward(varargin)
            % FeedForward creates a feedforward neural network object.
            % net = FeedForward(hiddenLayerSizes)
            % net = FeedForward(hiddenLayerSizes, isPatternNet)
            %
            % hiddenLayerSizes:     vector of layer sizes, e.g. [10 2 10]
            %                       describes three hidden layers with 10
            %                       neurons in the first, 2 neurons in the
            %                       second and 10 neurons in the third layer
            % isPatternNet:         if true, exit node converts values in range [0...1
            if nargin > 2 
               error('wrong number of input parameters') 
            elseif nargin == 2
                net.isPatternNet = varargin{2};
            end
            
            hiddenLayerSizes = varargin{1};
            net.initNetwork(hiddenLayerSizes);
            
            % -------------------------------------
            % define feedforward layer connections / currently unused
            % -------------------------------------
            net.biasConnect = ones(1, net.numLayers);
            
            net.inputConnect = zeros(1, net.numLayers);
            net.inputConnect(1) = 1; % input connects to layer 1 (inputConnect(layer))
            net.numInputs = sum(net.inputConnect);
            
            net.outputConnect = zeros(1, net.numLayers);
            net.outputConnect(net.numLayers) = 1; % last layer is set to output layer
            net.numOutputs = sum(net.outputConnect);
            
            net.layerConnect = zeros(net.numLayers);
            % connect each layer to its fellow layer (feedforward)
            for k = 2:net.numLayers
                net.layerConnect(k, k-1) = 1; % row = destination layer; column = source layer
            end
        end
        
        [y, a] = simulate(net, varargin)
        
        [E, g, output, jacobian] = train(net, input, target)
        
        configure(net, varargin)
        
        numWeights = getNumWeights(net)
        
        gradient = getGradientByWeight(net, gradVector, weight)
        
        weightVector = getWeightVector(net)
        
        setWeights(net, weights)
    end
    
end

