function [ abortFcn ] = makeAbortFcn( net, validateValues, validateTargets )
%ABORT Summary of this function goes here
%   Detailed explanation goes here

    abortFcn = @abort;

    function stop = abort(x,optimValues,state)
        stop = false;
                
        switch state
            case 'init'
                % do nothing
            case 'iter'
                
                [y, ~] = simulate(net, validateValues, false);

                Q = length(validateValues);
                E = 0;
                for q = 1:Q
                    E = E + nnfw.Util.mse(y(:, q), validateTargets(:, q));
                end
                
                if E < 1e-4
                    stop = true;
                    disp(['aborted training: E = ' num2str(E)]);
                end
                
            case 'done'
                % do nothing
            otherwise
        end
    end
end

