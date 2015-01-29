function plotSISO( data )
%PLOTSISO Plots the data prepared for SISO training method
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
%   size:               data size of the input signals
%   numInputs:          number of input signals
%   numTest:            number of test singals
%   y:                  net output of input signals (net(p))
%   yTest:              net output of test input signals (net(testP))
%   yExtra:             net output of extrapolation input signals (net(extraP))
%   xAxis:              data for the plot's x-axis 

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
            plotInput = plot(xData, y((k-1)*dataSize+1:k*dataSize), 'Color', colorInput, 'LineWidth', lwInput, 'Tag', 'input');
        end
        % plot simulated data from test data inputs
        for k = 1:numTest
            plotTestInput = plot(xData, yTest((k-1)*dataSize+1:k*dataSize), 'Color', colorTestInput, 'LineWidth', lwTestInput, 'Marker', 'x', 'Tag', 'testInput');
        end
        % plot extraploation data
        plotExtraInput = plot(xData, yExtra, 'Color', colorExtraInput, 'LineWidth', lwExtraInput, 'Tag', 'extraInput');
        legend([plotInput(1), plotTestInput(1), plotExtraInput], lgInput, lgTestInput, lgExtraInput);
    hold off;
    
end