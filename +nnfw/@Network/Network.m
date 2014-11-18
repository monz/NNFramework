classdef (Abstract) Network < handle
    %NET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        minmaxInputSettings;
        minmaxTargetSettings;
        
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
        initNetwork(obj, numInputs, numLayers, numOutputs);
        initWeights(obj) % set random layer weights for train purpose
        costFcn = makeCostFcn(net, fcn, input, target); % for fminunc
        costFcn = makeCostFcn2(net, fcn, input, target); % for lsqnonlin
    end
    
    methods (Abstract)
        configure(obj, varargin);
        simulate(obj, varargin);
        train(obj, input, target);
        getNumWeights(obj);
        getGradientByWeight(obj, gradVector, weight);
        getWeightVector(obj);
        setWeights(obj);
    end
    
end

