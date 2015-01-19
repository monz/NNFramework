function [ title, xLabel, yLabel ] = loadPlotData( idPtidC )
%LOADPLOTDATA Summary of this function goes here
%   Detailed explanation goes here

    % load data from .mat-file:
    load('clusteringData');
    
    title = clusteringData(idPtidC).partTypeName;
    xLabel = clusteringData(idPtidC).xLabel;
    yLabel = clusteringData(idPtidC).yLabel;

end

