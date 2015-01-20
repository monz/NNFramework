function plotSISO_Time( data )
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
    
    y = data.y;
    yTest = data.yTest;
    yExtra = data.yExtra;
    xData = data.xAxis';
    xData = xData(1,:);
    
    figure(figureNr);
    hold on
        title(titleText);
        xlabel(xLabel);
        ylabel(yLabel);
        % plot simulated data from trained inputs
        for k = 1:numInputs
            plotInput = plot(xData, y(1, (k-1)*dataSize+1:k*dataSize), 'Color', colorInput, 'LineWidth', lwInput, 'Tag', 'input');
        end
        % plot simulated data from test data inputs
        for k = 1:numTest
            plotTestInput = plot(xData, yTest(1, (k-1)*dataSize+1:k*dataSize), 'Color', colorTestInput, 'LineWidth', lwTestInput, 'Tag', 'testInput');
        end
        % plot extraploation data
        plotExtraInput = plot(xData, yExtra(1, :), 'Color', colorExtraInput, 'LineWidth', lwExtraInput, 'Tag', 'extraInput');
        legend([plotInput(1), plotTestInput(1), plotExtraInput], lgInput, lgTestInput, lgExtraInput);
    hold off;
    
end