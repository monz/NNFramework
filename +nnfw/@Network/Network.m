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
    end
    
    methods (Abstract)
        configure(obj, varargin);
        simulate(obj, input);
        train(obj, input, target);
        getNumWeights(obj);
        getGradientByWeight(obj, gradVector, weight);
    end
    
end

