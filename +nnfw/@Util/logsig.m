function x = logsig( n )
%LOGSIG Logarithmic sigmoid transfer function.

    x = 1 ./ (1 + exp(-n));

end

