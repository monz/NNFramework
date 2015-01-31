function dn = quadraticBackprop( x )
%QUADRATICBACKPROP is the derivation of the quadratic function.
%   This function is used in the backpropagation algorithm.
%
%   Don't use this function directly in neurons. Instead use
%   <a href="matlab:doc nnfw.Util.Activation">Activation</a>.
%   These Objects provide more information needed for internal use.

    dn = 2.*sqrt(x);
    
end

