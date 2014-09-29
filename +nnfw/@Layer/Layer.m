classdef Layer < handle
    %LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        type % input, hidden, output
        f % activation function
        size % number of layer neurons
    end
    
    methods
        function obj = Layer(layerType, transferFnc)
            obj.type = layerType;
            obj.f = transferFnc;
        end
    end
    
end

