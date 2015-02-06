function [ options ] = getOptionsString( data )
%GETOPTIONSSTRING Returns a string describing the options used in a neural network.
%   This string is used for the file names to save plots and workspace.
%
%   options = GETOPTIONSSTRING(data);
%
%   data.timeFlip:          timeFlip value set in the network
%   data.maxDimension:      maxDimension value set in the network
%   data.delay1:            delay1 value set in the network
%
%   Returns
%   options:                string describing the options used by the neural
%                           network

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

end

