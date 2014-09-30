function fitValue = nrmse( Actual, Expected )
%NRMSE Computes a fitValue measure for data.
%
%   fitValue = nrmse(Actual, Expected)
%   compares the data matrix 'Actual' with a reference value 'Expected' using
%   NRMSE as a comparison yardstick to produce the number fitValue. fitValue is a
%   quantitative representation of the closeness of 'Actual' to 'Expected'.
%
%   Input arguments:
%      Actual: Test data matrix with Ns samples and N channels; Ns-by-N
%              matrix. 'Actual' must not contain NaNs or Infs.
%    Expected: Reference data of same size as 'Actual'. Must not contain NaNs or
%              Infs.
%
%    The output argument fitValue is a row vector with N entries, where
%    N = size(Actual,2). NRMSE vary between -Inf (bad fit) to 1 (perfect fit);
%    a zero value for NRMSE denotes that the data 'Actual' is no
%    better than a straight line at matching 'Expected'.
%
%  Comparing multiple data sets:
%    Either 'Actual' or 'Expected' or both can be a cell array of double matrices if
%    you want to compare one or more test data sets to one or more
%    reference values. In that case, fitValue is a double array of size equal to
%    [Nx Cell_Array_Size], where Nx = N. If both 'Expected' are cell arrays, their sizes
%    must match so that fitValue(:,i1,i2,...) corresponds to Actual{i1,i2,...} and
%    Expeted{i1, i2,...}.

% 'NRMSE': fitValue(i) = 1 - (norm(Expected(:,i)-Actual(:,i)))/(norm(Expected(:,i)-mean(Expected(:,i))))

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
    fitValue = zeros(size(Expected{1},2),Ne); % N-by-Ne
    % compute nrmse
    for ct = 1:Ne
        xActual = Actual{min(ct,end)};
        xExpected = Expected{min(ct,end)};
        N = size(xActual,2);
        for k = 1:N
            numerator = norm(xExpected(:,k)-xActual(:,k));
            denominator = norm(xExpected(:,k)-mean(xExpected(:,k)));
            if denominator == 0 && numerator == 0
               temp = 0;
            else
               temp = numerator/denominator;
            end
            fitValue(k,ct) = 1 - temp;
        end
    end
end