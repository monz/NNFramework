function configure(net, varargin)
        % expected parameter =  net, input, traget
        % input and target have to be cell arrays
        % x-th row in input/target cell array defines x-th input/output
        if nargin < 3
            error('not enough arguments');
        elseif nargin > 3
            error('to much arguments');
        end

        % -------------------------------------
        % define network dimensions
        % -------------------------------------
        input = varargin{1};
        % extract min/max value information from input data
        [~,net.minmaxInputSettings] = nnfw.Util.minmaxMapping(input);
        % for further use convert input to cell array
        if ~iscell(input)
            input = {input};
        end

        % check given input size against network input size
        if size(input,1) ~= net.numInputs
            error('input parameter dimension missmatch');
        end
        % define input sizes
        for k = 1:net.numInputs
            net.inputs{k}.size = size(input{k,:},1);
        end

        output = varargin{2};
        % extract min/max value information from target data
        [~,net.minmaxTargetSettings] = nnfw.Util.minmaxMapping(output);
        % for further use convert output to cell array
        if ~iscell(output)
            output = {output};
        end

        % check given target size against network output size
        if size(output,1) ~= net.numOutputs
            error('output parameter dimension missmatch');
        end
        % define output sizes
        for k = 0:net.numOutputs-1
            net.outputs{net.numLayers+k}.size = size(output{k+1,:},1);
        end
end