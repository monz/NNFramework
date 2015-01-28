function [ out ] = daviesBouldin( in1, in2 )
%DAVIESBOULDIN  Value to measure the distance of two signals
%   Simple implementation of http://en.wikipedia.org/wiki/Davies-Bouldin_index
%   d = DAVIESBOULDIN(A,B) calculates the distance between signal A and signal B

    N = length(in1);

    sumResult = 0;
    for k = 1:N
        sumResult = sumResult + (in1(k)-in2(k))^2;
    end
    
    out = sqrt(1/N * sumResult);

end

