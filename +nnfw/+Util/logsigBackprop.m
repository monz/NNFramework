function dn = logsigBackprop( a )
%LOGSIGBACKPROP Summary of this function goes here
%   Detailed explanation goes here

    dn = (1-a).*(a);

end

