classdef Util
    %UTIL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Static)

        % header for logsig function, implementation in logsig.m
        x = logsig(n);
        
        % header for linear function, implementation in linear.m
        x = linear(n);

        % header for minmaxMappingRevert function, implementation in
        % minmaxMappingRevert.m
        output = minmaxMappingRevert( input, settings );
        
        % header for minmaxMapping function, implementation in
        % minmaxMapping.m
        [output, settings] = minmaxmapping(input, varargin);
    end
end

