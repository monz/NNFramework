clear all;
clc;

import matlab.unittest.TestSuite;

suite = TestSuite.fromPackage('nnfw.Test','IncludingSubpackages',true);
result = run(suite)
