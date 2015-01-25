function savePlot( data, plotData, plotOrigData, plotType )
%SAVEPLOT Summary of this function goes here
%   Detailed explanation goes here

    if ~exist(data.outDir, 'dir')
        mkdir(data.outDir);
    end

    close all;

    data.plotSpecific(plotData);
    data.plotCommon(plotOrigData);

    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf,'units','normalized','outerposition',[0 0 1 1]);
    
    formatString = '%s/%d_%s_%s_%s_%s_%d_%d%s_%s_%s.%s';
    
    netSize = '';
    for k = 1:length(data.numNeurons)
        if k == length(data.numNeurons)
            netSize = strcat(netSize,sprintf('%d', data.numNeurons(k)));
        else
            netSize = strcat(netSize,sprintf('%d-', data.numNeurons(k)));
        end
    end

    options = '';
    if data.timeFlip
        options = sprintf('_tf-%d', data.timeFlip);
    end
    if data.maxDimension > 0
        options = strcat(options,sprintf('_md-%d', data.maxDimension));
    end
    if length(data.delay1) > 1
        options = strcat(options,sprintf('_d1-%d-%d', data.delay1(1), data.delay1(end)));
    end

    for k = 1:length(data.ext)
        saveas(gcf, sprintf(formatString, ...
            data.outDir, data.idPtidC, data.netType, ...
            data.tb1, data.tb2, netSize, data.meanInput, data.meanTarget, ...
            options, plotType, data.date, data.ext{k}));
    end

end
