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
        backprop % derivated activation function handle
    end
    
    methods
        function obj = Activation(activationFunction, derivateActivationFunction)
                obj.f = activationFunction;
                obj.backprop = derivateActivationFunction;
        end
    end
    
    enumeration
       TANH (@tanh, @nnfw.Util.tanhBackprop)
       LOGSIG (@nnfw.Util.logsig, @nnfw.Util.logsigBackprop)
       PURELIN (@nnfw.Util.linear, @nnfw.Util.linearBackprop)
       QUAD (@nnfw.Util.quadratic, @nnfw.Util.quadraticBackprop)
    end
    
end

