function [ output ] = minmaxMappingApply( input, settings )
%MINMAXMAPPINGAPPLY Maps matrix values to min/max value based on the given settings.
%   Attention the mapping is row bases!
%
%   [output] = minmaxMapping(input, settings)
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