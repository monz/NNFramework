function weights = getRangedRandomWeights( numWeights, range )
%GETRANGEDRANDOMWEIGHTS Summary of this function goes here
%   Detailed explanation goes here
    maxValue = max(range);
    minValue = min(range);
    
    weights = (maxValue - minValue).*rand(numWeights,1) + minValue;
end

