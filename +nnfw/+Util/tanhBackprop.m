function dn = tanhBackprop( a )
%TANHBACKPROP is a slightly adapted version of the derivation of the tanh function
%   The original derivation is:    1 - tanh^2(x)
%   This function is used in the backpropagation algorithm.
%
%   Don't use this function directly in neurons. Instead use
%   <a href="matlab:doc nnfw.Util.Activation">Activation</a>.
%   These Objects provide more information needed for internal use.

    dn = 1 - a.*a;

end

