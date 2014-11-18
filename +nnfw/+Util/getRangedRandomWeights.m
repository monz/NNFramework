function weights = getRangedRandomWeights( numWeights, range )
%GETRANGEDRANDOMWEIGHTS Summary of this function goes here
%   Detailed explanation goes here

    maxAbsValue = max(abs(range));
    maxValue = max(range);
    minValue = min(range);

    weights = zeros(numWeights, 1);
    for k = 1:numWeights
        randValue = (rand()*2 - 1) * maxAbsValue;
        
        if randValue < minValue
            randValue = randValue + abs(maxAbsValue - minValue);
        elseif randValue > maxValue
            randValue = randValue - abs(maxAbsValue - maxValue);
        end        
        weights(k) = randValue;
    end
end

