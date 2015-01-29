classdef Layer < handle
    %LAYER Describes the layers of a neural network
    %   Each neuron has a activation function. In this implementation all
    %   neurons in the same layer have the same activation function f. The
    %   type of the network is used for clearness only. The size describes
    %   the total number of neurons contained in a layer.
    
    properties
        type % can be one of {'input', 'hidden', 'output'}
        f % activation function handle
        size % number of layer neurons
    end
    
    methods
        function obj = Layer(layerType, transferFnc)
            % Creates a layer with specified type and activation function
            obj.type = layerType;
            obj.f = transferFnc;
        end
    end
    
end

