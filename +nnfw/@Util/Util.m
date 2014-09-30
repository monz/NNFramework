classdef Util
    %UTIL Implements network activation and data preprocessing functions.
    %   Activation functions calculate the neurons exit value. Basic
    %   functions are log-sigmoid "logsig" or linear. Preprocessing
    %   functions prepare the network input data for better
    %   network training performance.
    %   Don't use this functions directly in neurons. Instead use
    %   <a href="matlab:doc nnfw.Activation">Activation</a>. These Objects
    %   provide more information needed for internal use.
    
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
        [output, settings] = minmaxMapping(input, varargin);
    end
end

