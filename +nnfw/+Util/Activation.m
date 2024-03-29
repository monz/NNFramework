classdef Activation
    %ACTIVATION Defines activation functions for direct use in neurons of neural netwroks
    %   Activation objects combine executable function handles with
    %   additional information needed for derivative calculation. The
    %   functions will be implemented in <a href="matlab:doc nnfw.Util">Util</a> package.
    %
    %   See also Util
    
    properties (SetAccess = immutable)
        f % activation function handle
        backprop % derivated activation function handle
        valueRange % value range for the layer weight initialization
    end
    
    methods
        function obj = Activation(activationFunction, derivateActivationFunction, valueRange)
                obj.f = activationFunction;
                obj.backprop = derivateActivationFunction;
                obj.valueRange = valueRange;
        end
    end
    
    enumeration
       TANH (@tanh, @nnfw.Util.tanhBackprop, [-1.5 1.5])
       LOGSIG (@nnfw.Util.logsig, @nnfw.Util.logsigBackprop, [-1.5 1.5])
       PURELIN (@nnfw.Util.linear, @nnfw.Util.linearBackprop, [-1 1])
       QUAD (@nnfw.Util.quadratic, @nnfw.Util.quadraticBackprop, [-1 1])
    end
    
end

