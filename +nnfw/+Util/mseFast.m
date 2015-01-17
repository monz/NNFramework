function fitValue = mseFast( Actual, Expected )
%MSE Computes a fitValue measure for data.
%
%   fitValue = mse(Actual, Expected)
%   compares the data matrix 'Actual' with a reference value 'Expected' using
%   MSE as a comparison yardstick to produce the number fitValue. fitValue is a
%   quantitative representation of the closeness of 'Actual' to 'Expected'.
%
%   Input arguments:
%      Actual: Test data matrix with Ns samples and N channels; Ns-by-N
%              matrix. 'Actual' must not contain NaNs or Infs.
%    Expected: Reference data of same size as 'Actual'. Must not contain NaNs or
%              Infs.
%
%   The output argument fitValue is a double scalar.
%
%  Comparing multiple data sets:
%   Either 'Actual' or 'Expected' or both can be a cell array of double matrices if
%   you want to compare one or more test data sets to one or more
%   reference values. In that case, fitValue is a double array of size equal to
%   [1 Cell_Array_Size]. If both 'Expected' are cell arrays, their sizes
%   must match so that fitValue(:,i1,i2,...) corresponds to Actual{i1,i2,...} and
%   Expected{i1, i2,...}.

%   'MSE': fitValue = norm(Actual-Expected)^2/(Ns-1);

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
    for ct = 1:Ne
        xActual = Actual{min(ct,end)};
        xExpected = Expected{min(ct,end)};
        C = bsxfun(@minus, xExpected,xActual);
        C = bsxfun(@times, C,C);
        E = sum(C(:));
        fitValue(ct) = E/size(xActual,1);
    end
end