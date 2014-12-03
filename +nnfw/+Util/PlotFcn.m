classdef PlotFcn
    %PLOTFUNCTION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = immutable)
        f % plot function handle
    end
    
    methods
        function obj = PlotFcn(plotFcn)
            obj.f = plotFcn;
        end
    end
    
    enumeration
        FVAL(@optimplotfval)
        STEPSIZE(@optimplotstepsize)
        X(@optimplotx)
        FIRSTORDEROPT(@optimplotfirstorderopt)
    end
end

