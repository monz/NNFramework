function [ output ] = minmaxMappingApply( input, settings )
%MINMAXMAPPING Maps matrix values to given min/max value.
%   Each row in a matrix get mapped to given min/max value, default [-1 1].
%
%   [output, settings] = minmaxMapping(input)
%   [output, settings] = minmaxMapping(input, min, max)
%
%   To revert this process use <a href="matlab:doc nnfw.Util.minmaxMappingRevert">minmaxMappingRevert</a> function.
%   Provide the settings returned from minmaxMapping to the revert function
%   to properly revert the process.
%
%   See also MINMAXMAPPINGREVERT


    if settings.no_change
      output = input;
      return;
    end

    output = bsxfun(@minus,input,settings.inOffset);
    output = bsxfun(@times,output,settings.gain);
    output = bsxfun(@plus,output,settings.outMin);
end