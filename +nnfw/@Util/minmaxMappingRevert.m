function output = minmaxMappingRevert( input, settings )
%MINMAXMAPPINGREVERT Summary of this function goes here
%   Detailed explanation goes here

    if settings.no_change
      output = y;
      return;
    end

    output = bsxfun(@minus,input,settings.outMin);
    output = bsxfun(@rdivide,output,settings.gain);
    output = bsxfun(@plus,output,settings.inOffset);    
end

