%%% -----------------------------------------------------------------------
%   This script starts the unit test test environment.
%   All tests in the +Test package of the NNFW will be executed. The
%   results will be printed to the command window.
%%% -----------------------------------------------------------------------

clear;
clc;

import matlab.unittest.TestSuite;

suite = TestSuite.fromPackage('nnfw.Test','IncludingSubpackages',true);
result = run(suite)
