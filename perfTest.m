clear;
clc;
import matlab.unittest.TestSuite;

numTestRuns = 10;
runtimes = zeros(numTestRuns,1);
for k = 1:numTestRuns
    suite = matlab.unittest.TestSuite.fromMethod(?performanceTest, 'performanceTestMIMO_01');
    result = run(suite);
    runtimes(k) = result.Duration;
end