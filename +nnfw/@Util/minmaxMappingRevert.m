function output = minmaxMappingRevert( input, settings )
%MINMAXMAPPINGREVERT Reverts the process of <a href="matlab:doc
%nnfw.minmaxMapping">minmaxMapping</a>
%   Each row in a matrix get mapped to its original min/max value specified
%   in the settings parameter.
%
%   output = minmaxMappingRevert( input, settings )
%
%   See also MINMAXMAPPING

    if settings.no_change
      output = y;
      return;
    end

    output = bsxfun(@minus,input,settings.outMin);
    output = bsxfun(@rdivide,output,settings.gain);
    output = bsxfun(@plus,output,settings.inOffset);    
end

