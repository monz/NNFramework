function plotSISO_NARX_Time( data )
%PLOTSISO Plots the data prepared for SISO_NARX-Delay training method
%   Each training method preprocesses the training data differently.
%   Therefore exists different plot functions which handle the data
%   specific to their preparation.
%
%   data is of type struct and should contain at least following
%   information:
%
%   figureNr:           figure handle number
%   title:              title of the plot
%   xLabel:             label of the x-axis 
%   yLabel:             label of the y-axis
%   lgInput:            legend of the input signals (p)
%   lgTestInput:        legend of the test input signals (testP)
%   lgExtraInput:       legend of the extrapolation input signals (extraP)
%   lwInput:            line width of the input singals
%   lwTestInput:        line width of the test input signals
%   lwExtraInput:       line width of the extrapolation input signals
%   colorInput:         line color of the input signals
%   colorTestInput:     line color of the test input signals
%   colorExtraInput:    line color of the extrapolation input signals
%   shift:              number of shifted values, e.g. delay 1:10 => shift = 10
%   size:               data size of the input signals
%   numInputs:          number of input signals
%   numTest:            number of test singals
%   y:                  net output of input signals (net(p))
%   yTest:              net output of test input signals (net(testP))
%   yExtra:             net output of extrapolation input signals (net(extraP))
%   xAxis:              data for the plot's x-axis 
%   flipTime:           true if signals have been flipped from left to right
%

    figureNr = data.figureNr;
    titleText = data.title;
    xLabel = data.xLabel;
    yLabel = data.yLabel;
    lgInput = data.lgInput;
    lgTestInput = data.lgTestInput;
    lgExtraInput = data.lgExtraInput;
    
    lwInput = data.lwInput;
    lwTestInput = data.lwTestInput;
    lwExtraInput = data.lwExtraInput;
    colorInput = data.colorInput;
    colorTestInput = data.colorTestInput;
    colorExtraInput = data.colorExtraInput;
    
    shift = data.shift;
    dataSize = data.size;
    dataSizeShifted = dataSize-shift;
    numInputs = data.numInputs;
    numTest = data.numTest;
    
    y = data.y;
    yTest = data.yTest;
    yExtra = data.yExtra;
    xData = data.xAxis(1:dataSize);
    
    flipTime = data.flipTime;
    
    figure(figureNr);
    hold on
        title(titleText);
        xlabel(xLabel);
        ylabel(yLabel);
        % plot simulated data from trained inputs
        
        if flipTime
            xStart = 1;
            xEnd = dataSize;
            yEnd = 0;
            for k = 1:numInputs
                if k == numInputs
                    xEnd = dataSizeShifted;
                    yEnd = dataSizeShifted;
                    yStart = (yEnd - dataSizeShifted) + 1;
                else
                    yEnd = yEnd + dataSize;
                    yStart = (yEnd - dataSize) + 1;
                end
                plotInput = plot(xData(xStart:xEnd), y(1, yStart:yEnd), 'Color', colorInput, 'LineWidth', lwInput, 'Tag', 'input');
            end
        else
            xEnd = dataSize;
            for k = 1:numInputs
                if k == 1
                    xStart = shift + 1;
                    yEnd = dataSize - shift;
                    yStart = 1;
                else
                    xStart = (xEnd - dataSize)+1;
                    yEnd = yEnd + dataSize;
                    yStart = (yEnd - dataSize) + 1;
                end
                plotInput = plot(xData(xStart:xEnd), y(1, yStart:yEnd), 'Color', colorInput, 'LineWidth', lwInput, 'Tag', 'input');
            end
        end
        % plot simulated data from test data inputs
        if flipTime
            xStart = 1;
            xEnd = dataSizeShifted;
            for k = 1:numTest
                yEnd = k*dataSizeShifted;
                yStart = (yEnd - dataSizeShifted) + 1;
                plotTestInput = plot(xData(xStart:xEnd), yTest(1, yStart:yEnd), 'Color', colorTestInput, 'LineWidth', lwTestInput, 'Tag', 'testInput');
            end
        else
            xStart = shift + 1;
            for k = 1:numTest
                yEnd = k*dataSizeShifted;
                yStart = (yEnd - dataSizeShifted) + 1;
                plotTestInput = plot(xData(xStart:xEnd), yTest(1, yStart:yEnd), 'Color', colorTestInput, 'LineWidth', lwTestInput, 'Tag', 'testInput');
            end
        end
        % plot extraploation data
        if flipTime
            plotExtraInput = plot(xData(1:dataSizeShifted), yExtra(1, :), 'Color', colorExtraInput, 'LineWidth', lwExtraInput, 'Tag', 'extraInput');
        else
            plotExtraInput = plot(xData(shift+1:dataSize), yExtra(1, :), 'Color', colorExtraInput, 'LineWidth', lwExtraInput, 'Tag', 'extraInput');
        end
        
        legend([plotInput(1), plotTestInput(1), plotExtraInput], lgInput, lgTestInput, lgExtraInput);
    hold off;
    
end