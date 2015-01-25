function plotSISO_NARX_Time( data )
%PLOTSISO Summary of this function goes here
%   Detailed explanation goes here

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