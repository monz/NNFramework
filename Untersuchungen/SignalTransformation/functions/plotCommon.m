function plotCommon( data )
%PLOTCOMMON   Plots the original data of test bench 1 and 2
%   This is a helper function to plot the test bench's original data. It
%   combines the plots of the given figure with the additional data.
%
%   data is of type struct and should contain at least following
%   information:
%
%   xDataTB1:           all data for test bench one x-axis
%   xDataTB2:           all data for test bench two x-axis
%   yDataTB1:           all data for test bench one y-axis
%   yDataTB2:           all data for test bench two y-axis
%   yDataMeanTB1:       mean data for test bench one y-axis
%   yDataMeanTB2:       mean data for test bench two y-axis
%   lgTB1:              legend for test bench one
%   lgTB2:              legend for test bench tow
%   lgInput:            legend for input signals
%   lgTestInput:        legend for test input signals
%   lgExtraInput:       legend for extrapolation signals
%   colorTB1:           line color for test bench one
%   colorTB2:           line color for test bench two
%   colorMeanTB1:       line color for test bench one mean data
%   colorMeanTB2:       line color for test bench two mean data
%   lineStyleMeanTB1:   line style for test bench one mean data
%   lineStyleMeanTB2:   line style for test bench two mean data
%   meanOnly:           if true only tb1's and tb2's mean data gets plotted
%   referenceOnly:      if true only the reference values gets plotted, reference value is always tb2
%   figureNr:           figure handle number
%

        xDataTB1 = data.xAxisTB1;
        xDataTB2 = data.xAxisTB2;
        yDataTB1 = data.yAxisTB1;
        yDataTB2 = data.yAxisTB2;
        yDataMeanTB1 = data.yAxisMeanTB1;
        yDataMeanTB2 = data.yAxisMeanTB2;
        
        lgTB1 = data.lgTB1;
        lgTB2 = data.lgTB2;
        lgInput = data.lgInput;
        lgTestInput = data.lgTestInput;
        lgExtraInput = data.lgExtraInput;
        
        colorTB1 = data.colorTB1;
        colorTB2 = data.colorTB2;
        colorMeanTB1 = data.colorMeanTB1;
        colorMeanTB2 = data.colorMeanTB2;
        lineStyleMeanTB1 = data.lineStyleMeanTB1;
        lineStyleMeanTB2 = data.lineStyleMeanTB2;
        
        meanOnly = data.meanOnly;
        referenceOnly = data.referenceOnly;
        figureNr = data.figureNr;
        
        
        % extract line handles from former line plotting
        inputLine = findobj('Tag','input');
        testInputLine = findobj('Tag','testInput');
        extraInputLine = findobj('Tag','extraInput');
        % plot data
        figure(figureNr)
        hold on
            if ~meanOnly
                if referenceOnly
                    TB2Plot = plot(xDataTB2, yDataTB2, 'Color', colorTB2);
                else
                    TB1Plot = plot(xDataTB1, yDataTB1, 'Color', colorTB1);
                    TB2Plot = plot(xDataTB2, yDataTB2, 'Color', colorTB2);
                end
            end
            
            TB1MeanPlot = plot(xDataTB1, yDataMeanTB1, 'Color', colorMeanTB1, 'LineStyle', lineStyleMeanTB1, 'LineWidth', 2);
            TB2MeanPlot = plot(xDataTB2, yDataMeanTB2, 'Color', colorMeanTB2, 'LineStyle', lineStyleMeanTB2,'LineWidth', 2);

            if meanOnly
                legend([TB1MeanPlot(1), TB2MeanPlot(1), inputLine(1), testInputLine(1), extraInputLine], strcat('Mean ', lgTB1), strcat('Mean ', lgTB2), lgInput, lgTestInput, lgExtraInput);
            else
                if referenceOnly
                    legend([TB2Plot(1), TB1MeanPlot(1), TB2MeanPlot(1), inputLine(1), testInputLine(1), extraInputLine], lgTB2, strcat('Mean ', lgTB1), strcat('Mean ', lgTB2), lgInput, lgTestInput, lgExtraInput);
                else
                    legend([TB1Plot(1), TB2Plot(1), TB1MeanPlot(1), TB2MeanPlot(1), inputLine(1), testInputLine(1), extraInputLine], lgTB1, lgTB2, strcat('Mean ', lgTB1), strcat('Mean ', lgTB2), lgInput, lgTestInput, lgExtraInput);
                end
            end
        hold off
end

