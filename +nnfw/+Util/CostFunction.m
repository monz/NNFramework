classdef CostFunction
    %COSTFUNCTION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = immutable)
        f
    end
    
    methods
        function obj = CostFunction(costFunction)
           obj.f = costFunction; 
        end
    end
    
    enumeration
       MSE(@nnfw.Util.mse);
       NRMSE(@nnfw.Util.nrmse);
       NMSE(@nnfw.Util.nmse);
       COMPONENTERROR(@nnfw.Util.componentError);
    end
end

