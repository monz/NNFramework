function [ out ] = daviesBouldin( in1, in2 )
%DAVIESBOULDEN Summary of this function goes here
%   Detailed explanation goes here

    N = length(in1);

    sumResult = 0;
    for k = 1:N
        sumResult = sumResult + (in1(k)-in2(k))^2;
    end
    
    out = sqrt(1/N * sumResult);

end

