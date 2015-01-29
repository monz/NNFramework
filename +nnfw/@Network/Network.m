classdef (Abstract) Network < handle
    %NETWORK Contains common attributes of all neural network types.
    %   NETWORK is a abstract class, hence no objects of type NETWORK can
    %   be created. Instead create objects of one of the subclasses. E.g.
    %   FeedForward. It implements a special version of the NETWORK class.
    %   The NETWORK class extends the handle class. This means if a network
    %   object (or subclass of it) is passed to a function, this function can
    %   change the attributes directly, because the object is not copied on
    %   function call.
    %
    %   See also HANDLE, FEEDFORWARD.
    
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
        
        IW = {} % input weights for each input layer
        LW = {} % layer weights for each hidden layer
        b = [] % biases
        
    end
    
    methods
        initNetwork(obj, hiddenLayerSizes); % initializes the network layers with default values
        initWeights(obj) % set random layer weights for train purpose
        costFcn = makeCostFcn(net, fcn, input, target); % for fminunc
        costFcn = makeCostFcn2(net, fcn, input, target); % for lsqnonlin
        
        function obj = Network()
            % Initializes the default optimization values on object
            % creation
            initOptimValues(obj)
        end
        
        function initOptimValues(obj) 
            % initialize optimazation values with defaults
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
        % The implementation should configure the network layer sizes properly.
        configure(obj, varargin);
        % The implementation should simulate the values through the network while considering the network characteristics.
        simulate(obj, varargin);
        % The implementation should train the adapt the network weights while considering the network characteristics.
        train(obj, input, target);
        % The implementation should return the total number of weight values in the neural network.
        getNumWeights(obj);
        % The implementation should return the gradient value related to the given weight value.
        getGradientByWeight(obj, gradVector, weight);
        % The implementation should return all weight values in a single vector
        getWeightVector(obj);
        % The implementation should replace the networks weight vector with the giben weight vector
        setWeights(obj, weights);
    end
    
end

