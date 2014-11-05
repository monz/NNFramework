classdef FeedForward < nnfw.Network
    %FEEDFORWARD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
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
        
        [y, a] = simulate(net, input)
        
        [E, g] = train(net, input, target)
        
        configure(net, varargin)
        
        numWeights = getNumWeights(net)
        
        gradient = getGradientByWeight(net, gradVector, weight)
        
        weightVector = getWeightVector(net)
        
        setWeights(net, weights)
    end
    
end

