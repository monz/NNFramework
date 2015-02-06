function [ netSize ] = getNetSizeString( numNeurons )
%GETNETSIZESTRING Returns a string describing the neural network's layer sizes.
%   This string is used for the file names to save plots and workspace.
%
%   netSize = GETNETSIZESTRING(data);
%
%   numNeurons:         vector describing the number of neurons and layers
%                       in the neural network
%
%   Returns
%   netSize:            string describing the network's size

    netSize = '';
    for k = 1:length(numNeurons)
        if k == length(numNeurons)
            netSize = strcat(netSize,sprintf('%d', numNeurons(k)));
        else
            netSize = strcat(netSize,sprintf('%d-', numNeurons(k)));
        end
    end

end

