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
    
    dataSize = data.size;
    numInputs = data.numInputs;
    numTest = data.numTest;
    shift = data.shift;
    
    y = data.y;
    yTest = data.yTest;
    yExtra = data.yExtra;
    xData = data.xAxis(1:dataSize);
    
    figure(figureNr);
    hold on
        title(titleText);
        xlabel(xLabel);
        ylabel(yLabel);
        % plot simulated data from trained inputs
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
        % plot simulated data from test data inputs
        for k = 1:numTest
            if k == 1
                xStart = shift + 1;
                yEnd = dataSize - shift;
                yStart = 1;
            else
                xStart = (xEnd - dataSize)+1;
                yEnd = yEnd + dataSize;
                yStart = (yEnd - dataSize) + 1;
            end
            plotTestInput = plot(xData(xStart:xEnd), yTest(1, yStart:yEnd), 'Color', colorTestInput, 'LineWidth', lwTestInput, 'Tag', 'testInput');
        end
        % plot extraploation data
        plotExtraInput = plot(xData(shift+1:dataSize), yExtra(1, :), 'Color', colorExtraInput, 'LineWidth', lwExtraInput, 'Tag', 'extraInput');
        
        legend([plotInput(1), plotTestInput(1), plotExtraInput], lgInput, lgTestInput, lgExtraInput);
    hold off;
    
end