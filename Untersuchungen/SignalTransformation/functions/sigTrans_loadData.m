function [ tb_data ] = sigTrans_loadData( idPtidC, test_bench, axis )
    % load data from .mat-file:
    load('clusteringData');
    
    % data indexes corresponding to given test bench
    dataindexTB = ismember(clusteringData(idPtidC).testbench, test_bench);
    % load data
    if axis == 'x'
        tb_data = clusteringData(idPtidC).xData(:,dataindexTB);
    elseif axis == 'y'
        tb_data = clusteringData(idPtidC).yData(:,dataindexTB);
    else
        error('Either "x" or "y" axis');
    end

end