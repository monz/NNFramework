function plot_kt2_to_kt3( idPtidC, meanOnly )

    %% plot data KT2 to KT3
    % clear;
    % clc;
    % close all;

    %% prepare data
    % load signal transformation data
    sigTrans_loadData;

    % kt2
    dataindexKT2 = ismember(clusteringData(idPtidC).testbench, 'kt2');
    % kt3
    dataindexKT3 = ismember(clusteringData(idPtidC).testbench, 'kt3');

    % calculate mean signals
    yMeanKT2 = mean(clusteringData(idPtidC).yData(:, dataindexKT2), 2);
    yMeanKT3 = mean(clusteringData(idPtidC).yData(:, dataindexKT3), 2);

    %% plot data
    figure(2);
    hold on
        xlabel(clusteringData(idPtidC).xLabel);
        ylabel(clusteringData(idPtidC).yLabel);
        title(clusteringData(idPtidC).partTypeName);
        % plot data
        if ~meanOnly
            kt2Plot = plot(clusteringData(idPtidC).xData(:,dataindexKT2), clusteringData(idPtidC).yData(:,dataindexKT2), 'Color', [0.0 0.0 0.7]);
            kt3Plot = plot(clusteringData(idPtidC).xData(:,dataindexKT3), clusteringData(idPtidC).yData(:,dataindexKT3), 'Color', [0.0 0.7 0.0]);
        end
        % plot mean data
        kt2MeanPlot = plot(clusteringData(idPtidC).xData(:,dataindexKT2), yMeanKT2, 'm', 'LineWidth', 2);
        kt3MeanPlot = plot(clusteringData(idPtidC).xData(:,dataindexKT2), yMeanKT3, 'k', 'LineWidth', 2);
        if ~meanOnly
            legend([kt2Plot(1), kt3Plot(1), kt2MeanPlot(1), kt3MeanPlot(1)], 'KT2','KT3', 'Mean KT2', 'Mean KT3');
        else
            legend([kt2MeanPlot(1), kt3MeanPlot(1)], 'Mean KT2', 'Mean KT3');
        end
    hold off

end