classdef Activation
    %ACTIVATION Defines activation functions for direct use in neurons of
    %neural netwroks
    %   Activation objects combine executable function handles with
    %   additional information needed for derivative calculation. The
    %   functions will be implemented in <a href="matlab:doc nnfw.Util">Util</a> package.
    %
    %   See also Util
    
    properties (SetAccess = immutable)
        f % activation function handle
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

