function [ abortFcn ] = makeAbortFcn( net, values )
%MAKEABORTFCN Creates a wrapper function used in the optimization function lsqnonlin
%   The internal function evaluates and plots the training/validation/test
%   errors. This function checks the error values against the configured
%   abort thresholds and decides whether to abort the training or not. Also
%   it provides a stop button in the plot window. The user can stop the
%   training by clicking on that button.
%
%   [ abortFcn ] = MAKEABORTFCN( net, values )
%
%   net:        the neural network to be trained
%   values:     cellarray containing the training/validation/test data
%
%   Returns
%   abortFcn:    wrapper function used in a optimization function

    abortFcn = @abort;

    function stop = abort(x,optimValues,state)
        persistent lastE;
        persistent bestWeights;
        persistent increaseCounter;
        persistent myTitle;
        persistent legendString;
        persistent numTrainValues;
        
        trainValues = values{1,1};
        trainTargets = values{1,2};
        validateValues = values{2,1};
        validateTargets = values{2,2};
        testValues = values{3,1};
        testTargets = values{3,2};

        stop = false;

        % define push button callback function
        function cbStopFcn(hObject,eventdata,handles)
            net.optim.stopTraining = true;
        end
        
        switch state
            case 'init'
                legendString = {'TrainError', 'ValidationError','TestError'};
                increaseCounter = 0;
                lastE = intmax;
                numTrainValues = size(trainValues,2);
            case 'iter'
                % -------------------------
                % calculate training error
                % -------------------------
                [y, ~] = simulate(net, trainValues, false);
                ETraining = nnfw.Util.mseFast(y, trainTargets);
                if numTrainValues > 1
                    % -------------------------
                    % calculate validation error
                    % -------------------------
                    [y, ~] = simulate(net, validateValues, false);
                    EValidate = nnfw.Util.mseFast(y, validateTargets);
                    if EValidate < lastE
                        %fprintf('lastE %d; EValidate %d \n', lastE, EValidate);
                        lastE = EValidate;
                        bestWeights = x;
                    else
                        increaseCounter = increaseCounter + 1;
                        if increaseCounter >= net.optim.maxErrorIncrease;
                            stop = true;
                            fprintf('aborted on increasing validation error %d times; E = %d\n', increaseCounter, EValidate);
                            fprintf('reseting best weights');
                            net.setWeights(bestWeights);
                        end
                    end
                    if EValidate < net.optim.abortThreshold
                        stop = true;
                        sprintf('Validation error reached abort threshold, aborted training: E = %d\n', EValidate);
                    end
                    % -------------------------
                    % calculate test error
                    % -------------------------
                    [y, ~] = simulate(net, testValues, false);
                    ETest = nnfw.Util.mseFast(y, testTargets);
                else
                    EValidate = 0;
                    ETest = 0;
                end
                % -------------------------
                % plot error values
                % -------------------------
                figure(3);
                uicontrol('Style', 'pushbutton', 'String', 'stop', 'Callback', @cbStopFcn);
                if optimValues.iteration == 0
                    % The 'iter' case is  called during the zeroth iteration,
                    % but it now has values that were empty during the 'init' case
                    hold on
                    plotValue.tra = plot(optimValues.iteration, ETraining, '-gd', 'MarkerFaceColor',[1 0 1]);
                    plotValue.val = plot(optimValues.iteration, EValidate, '-kd', 'MarkerFaceColor',[1 0 1]);
                    plotValue.tes = plot(optimValues.iteration, ETest, '-rd', 'MarkerFaceColor',[1 0 1]);
                    xlabel('Iteration');
                    set(plotValue.tra,'Tag','trainError');
                    set(plotValue.val,'Tag','validateError');
                    set(plotValue.tes,'Tag','testError');
                    ylabel('MSE');
                    title(sprintf('Best ValidationErrorValue: %0.5e  in Iteration: %d', EValidate, optimValues.iteration));
                    myTitle = get(gca,'Title');
                    set(gca,'YScale','log');
                    legend(legendString);
                    hold off
                else
                    hold on
                    plotValue.tra = findobj(get(gca,'Children'),'Tag','trainError');
                    plotValue.val = findobj(get(gca,'Children'),'Tag','validateError');
                    plotValue.tes = findobj(get(gca,'Children'),'Tag','testError');
                    newX = [get(plotValue.val,'Xdata') optimValues.iteration];
                    newTrainY = [get(plotValue.tra,'Ydata') ETraining];
                    newValidateY = [get(plotValue.val,'Ydata') EValidate];
                    newTestY = [get(plotValue.tes,'Ydata') ETest];
                    set(plotValue.tra,'Xdata',newX, 'Ydata',newTrainY);
                    set(plotValue.val,'Xdata',newX, 'Ydata',newValidateY);
                    set(plotValue.tes,'Xdata',newX, 'Ydata',newTestY);
                    set(myTitle,'String', sprintf('Best ValidationErrorValue: %0.5e  in Iteration: %d', EValidate, optimValues.iteration));
                    hold off
                end
                % check if stop button was pushed
                if net.optim.stopTraining
                    stop = true;
                end
            case 'done'
                % do nothing
            otherwise
        end
    end
end

