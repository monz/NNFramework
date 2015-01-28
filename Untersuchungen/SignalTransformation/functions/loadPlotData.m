function [ title, xLabel, yLabel ] = loadPlotData( idPtidC )
%LOADPLOTDATA   Loads description data from data set
%   [title, xLabel, yLabel] = LOADPLOTDATA( ID ) loads description of test bench given by ID.

    % load all data from .mat-file:
    load('clusteringData');
    
    title = clusteringData(idPtidC).partTypeName;
    xLabel = clusteringData(idPtidC).xLabel;
    yLabel = clusteringData(idPtidC).yLabel;

end

