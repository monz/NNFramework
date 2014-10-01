classdef (Abstract) Network < handle
    %NET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numInputs = 0
        numLayers = 0
        numOutputs = 0
        
        inputs = {}
        layers = {}
        outputs = {}
        
        biasConnect = []
        inputConnect = []
        layerConnect = []
        outputConnect = []
        
        IW = {}
        LW = {}
        b = []
        
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
            end
            % output layer
            for k = 1:numOutputs
                obj.outputs{1,k} = nnfw.Layer('output', nnfw.Util.Activation.PURELIN);
            end
        end
    end
    
    methods (Abstract)
        configure(obj, varargin);
        simulate(obj, input);
    end
    
end
