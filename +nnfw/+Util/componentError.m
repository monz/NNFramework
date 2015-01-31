function fitValue = componentError( Actual, Expected )
%COMPONENTERROR Computes a fitValue measure for data.
%
%   fitValue = COMPONENTERROR(Actual, Expected)
%
%   subtracts the 'Expected' values from the 'Actual' values component by component.
%   fitValue is a quantitative representation of the closeness of 'Actual' to 'Expected'.
%
%   Input arguments:
%      Actual: Test data matrix with Ns samples and N channels; Ns-by-N
%              matrix. 'Actual' must not contain NaNs or Infs.
%    Expected: Reference data of same size as 'Actual'. Must not contain NaNs or
%              Infs.
%
%   The output argument fitValue is a double scalar.

    % convert input to cell array 
    if ~iscell(Expected)
        Expected = {Expected};
    end
    if ~iscell(Actual)
        Actual = {Actual};
    end
    % check if dimensions match, otherwise it is impossible to compare the
    % values
    errMsg = 'dimensions do not match';
    if ~isequal(size(Actual),size(Expected)) && ~isscalar(Actual) && ~isscalar(Expected)
       error(errMsg);
    else
       for ct = 1:numel(Actual)
          if ~isequal(size(Actual{min(ct,1)}),size(Expected{min(ct,1)}))
             error(errMsg);
          end
       end
    end

    Ne = length(Expected);
    fitValue = zeros(1,Ne);
    % compute component error
    for ct = 1:Ne
        xActual = Actual{min(ct,end)};
        xExpected = Expected{min(ct,end)};
        fitValue = xExpected - xActual;
    end
end