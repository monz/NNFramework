function plotCommon( data )
%PLOTCOMMON Summary of this function goes here
%   Detailed explanation goes here

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

