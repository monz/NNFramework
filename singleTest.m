clear;
clc;
import matlab.unittest.TestSuite;

% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.networkUtilTest, 'setWeights_02');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.networkUtilTest, 'setWeights_03');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.networkUtilTest, 'getWeightVectorTest_03');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.networkUtilTest, 'getWeightVectorTest_04');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.networkUtilTest, 'getNumWeightsTest_02');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.gradientTest, 'trainSingleValue');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.gradientTest, 'trainQValues');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.gradientTest, 'trainVectorValues');
% suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.simulateModelTest, 'simulate');
suite = matlab.unittest.TestSuite.fromMethod(?nnfw.Test.jacobianTest, 'jacobianSimpleTest_02');
result = run(suite)
