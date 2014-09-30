function [ output, settings ] = minmaxMapping( input, varargin )
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

    % outMin/outMax default settings
    outMin = -1;
    outMax = 1;
    % outMin/outMax user settings
    if (nargin == 3) && isnumeric(varargin{1}) && isnumeric(varargin{2})
        outMin = varargin{1};
        outMax = varargin{2};
    end

    inRows = size(input,1);
    if isempty(input)
        input = nan(inRows,1);
    end
    inMin = min(input,[],2);
    inMax = max(input,[],2);
    % set all inMin/inMax NaN to -inf/inf
    inMin(isnan(inMin)) = -inf;
    inMax(isnan(inMax)) = inf;

    % Assert: inMin and inMax will be [-inf inf] for unknown ranges
    settings.name = 'minmaxmapping';
    settings.inRows = inRows;
    settings.inMax = inMax;
    settings.inMin = inMin;
    settings.inRange = inMax - inMin;
    settings.outRows = settings.inRows;
    settings.outMax = outMax;
    settings.outMin = outMin;
    settings.outRange = settings.outMax - settings.outMin;

    % Convert from settings values to safe processing values
    % and check whether safe values result in input<->output change.
    inOffset = settings.inMin;
    gain = settings.outRange ./ settings.inRange;
    fix = find((abs(gain)>1e14) | ~isfinite(settings.inRange) | (settings.inRange == 0));
    gain(fix) = 1;
    inOffset(fix) = settings.outMin;
    settings.no_change = (settings.inRows == 0) || (all(gain == 1) && all(inMin == 0));

    settings.gain = gain;
    settings.inOffset = inOffset;

    if settings.no_change
      output = input;
      return;
    end

    output = bsxfun(@minus,input,settings.inOffset);
    output = bsxfun(@times,output,settings.gain);
    output = bsxfun(@plus,output,settings.outMin);
end