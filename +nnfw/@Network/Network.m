classdef (Abstract) Network < handle
    %NET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        isPatternNet = false; % if true, exit node converts values in range [0...1]
        
        optim = struct; % contains training abort thresholds
        
        minmaxInputSettings; % settings for min/max conversion
        minmaxTargetSettings; % settings for min/max conversion
        
        valueIndexes; % indexes of separated input/target values
        
        numInputs = 0 % total number of network input vectors
        numLayers = 0 % number of network hidden layers
        numOutputs = 0 % total number of network output vectors
        
        inputs = {} % input layer information
        layers = {} % hidden layer information
        outputs = {} % output layer information
        
        biasConnect = [] % currently unused
        inputConnect = [] % currently unused
        layerConnect = [] % currently unused
        outputConnect = [] % currently unused
        
        IW = {} % input weights for each input vector
        LW = {} % layer weights for each hidden layer
        b = [] % biases
        
    end
    
    methods
        initNetwork(obj, numInputs, numLayers, numOutputs);
        initWeights(obj) % set random layer weights for train purpose
        costFcn = makeCostFcn(net, fcn, input, target); % for fminunc
        costFcn = makeCostFcn2(net, fcn, input, target); % for lsqnonlin
        
        function obj = Network()
            initOptimValues(obj)
        end
        % initialize optimazation values with defaults
        function initOptimValues(obj) 
            obj.optim.plotFcns = {};
        
            obj.optim.vlFactor = 0.20; % validation data factor, default 20 percent of input data are used for validation
            obj.optim.tsFactor = 0.05; % test data factor, default 5 percent of input data are used for testing
            
            obj.optim.abortThreshold = 1e-2; % if validation error is below this value, training gets aborted
            obj.optim.maxErrorIncrease = 5; % if the validation error increases x times, training gets aborted
            
            obj.optim.stopTraining = false; % if true training get stopped
            obj.optim.maxIter = 1000; % max number of iterations in optimization

            obj.optim.minmaxMapping = true; % enable/disable value mapping
        end
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

