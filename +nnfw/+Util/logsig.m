function x = logsig( n )
%LOGSIG Logarithmic sigmoid transfer function.
% Don't use this function directly in neurons. Instead use
% <a href="matlab:doc nnfw.Util.Activation">Activation</a>. These Objects
% provide more information needed for internal use.

    x = 1 ./ (1 + exp(-n));

end

