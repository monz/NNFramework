classdef FeedForward < nnfw.Network
    %FEEDFORWARD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function net = FeedForward(hiddenLayerSizes)
            net.initNetwork(hiddenLayerSizes);
            
            % -------------------------------------
            % define feedforward layer connections
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
        
        [E, g, output, lambda, jacobian] = train(net, input, target)
        
        configure(net, varargin)
        
        numWeights = getNumWeights(net)
        
        gradient = getGradientByWeight(net, gradVector, weight)
        
        weightVector = getWeightVector(net)
        
        setWeights(net, weights)
    end
    
end

