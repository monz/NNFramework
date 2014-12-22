function [ abortFcn ] = makeAbortFcn( net, values )
%ABORT Summary of this function goes here
%   Detailed explanation goes here

    abortFcn = @abort;

    function stop = abort(x,optimValues,state)
        persistent lastETest
        persistent increaseCounter;
        persistent myTitle;

        legendString = {'TrainError', 'ValidationError','TestError'};

        lastETest = intmax;        
        stop = false;
        
        trainValues = values{1,1};
        trainTargets = values{1,2};
        validateValues = values{2,1};
        validateTargets = values{2,2};
        testValues = values{3,1};
        testTargets = values{3,2};
        
        % define push button callback function
        function cbStopFcn(hObject,eventdata,handles)
            net.optim.stopTraining = true;
        end
        
        switch state
            case 'init'
                % do nothing
            case 'iter'
                % -------------------------
                % calculate training error
                % -------------------------
                [y, ~] = simulate(net, trainValues, false);
                ETraining = nnfw.Util.mseFast(y, trainTargets);
                % -------------------------
                % calculate validation error
                % -------------------------
                [y, ~] = simulate(net, validateValues, false);
                EValidate = nnfw.Util.mseFast(y, validateTargets);
                if EValidate < lastETest
                    lastETest = EValidate;
                else
                    increaseCounter = increaseCounter + 1;
                    if increaseCounter > 3
                        stop = true;
                        disp(['aborted training: E = ' num2str(EValidate)]);
                    end
                end
                % -------------------------
                % calculate test error
                % -------------------------
                [y, ~] = simulate(net, testValues, false);
                ETest = nnfw.Util.mseFast(y, testTargets);
                if ETest < lastETest
                    lastETest = ETest;
                    % maybe store here best weight vector in e.g. net.bestWeights
                    % set net.bestWeights in at the bottom of train function
                else
                    increaseCounter = increaseCounter + 1;
                    if increaseCounter > net.optim.maxErrorIncrease
                        stop = true;
                        disp(['aborted training: ValidationError increased = ' num2str(increaseCounter) ' times']);
                    end
                end
                if ETest < net.optim.abortThreshold
                    stop = true;
                    disp(['aborted training: E = ' num2str(ETest)]);
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

