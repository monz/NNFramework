function weights = getRangedRandomWeights( numWeights, range )
%GETRANGEDRANDOMWEIGHTS Generates random values in the specified range.
%   The boundary values are very unlikely to be among the generated values.
%
%   weights = getRangedRandomWeights(numWeights, range)
%
%   numWeights:     number of values to genrate
%   range:          range in which the generated values should be,
%                   e.g. [-1 1] generates values from -1 to 1
%
%   Returns
%   weights:        random values in specified range

    maxValue = max(range);
    minValue = min(range);
    
    weights = (maxValue - minValue).*rand(numWeights,1) + minValue;
end

