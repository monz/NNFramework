function [ tb_data ] = sigTrans_loadData( idPtidC, test_bench, axis )
%SIGTRANS_LOADDATA Loads test bench data from data structure
%  
%   idPtidC:        id of the test
%   test_bench:     test bench, one of {'kt2', 'kt3', 'kt4'}
%   axis:           which axis to load, one of {'x','y'}

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