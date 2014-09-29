classdef Activation
    %ACTIVATION Defines activation functions used for neurons in neural
    %netwroks
    %   Detailed explanation goes here
    
    properties (SetAccess = immutable)
        f
    end
    
    methods
        function obj = Activation(activationFunction)
                obj.f = activationFunction;
        end
    end
    
    enumeration
       TANH (@tanh) 
       LOGSIG (@nnfw.Util.logsig)
       PURELIN (@nnfw.Util.linear)
    end
    
end

