function dy = tanhDerivated( y )
%TANHDERIVED Summary of this function goes here
%   Detailed explanation goes here

    dy = 1 - tanh(y) * tanh(y);

end

