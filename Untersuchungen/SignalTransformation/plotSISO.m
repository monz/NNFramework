function plotSISO( data )
%PLOTSISO Summary of this function goes here
%   Detailed explanation goes here

    figureNr = data.figureNr;
    titleText = data.title;
    xLabel = data.xLabel;
    yLabel = data.yLabel;
    lgInput = data.lgInput;
    lgTestInput = data.lgTestInput;
    
    lwInput = data.lwInput;
    lwTestInput = data.lwTestInput;
    colorInput = data.colorInput;
    colorTestInput = data.colorTestInput;
    
    dataSize = data.size;
    numInputs = data.numInputs;
    numTest = data.numTest;
    
    y = data.y;
    yTest = data.yTest;
    xData = data.xAxis';
    xData = xData(1,:);
    
    figure(figureNr);
    hold on
        title(titleText);
        xlabel(xLabel);
        ylabel(yLabel);
        % plot simulated data from trained inputs
        for k = 1:numInputs
            plotInput = plot(xData, y((k-1)*dataSize+1:k*dataSize), 'Color', colorInput, 'LineWidth', lwInput, 'Tag', 'input');
        end
        % plot simulated data from test data inputs
        for k = 1:numTest
            plotTestInput = plot(xData, yTest((k-1)*dataSize+1:k*dataSize), 'Color', colorTestInput, 'LineWidth', lwTestInput, 'Tag', 'testInput');
        end

        legend([plotInput(1), plotTestInput(1)], lgInput, lgTestInput);
    hold off;
    
end