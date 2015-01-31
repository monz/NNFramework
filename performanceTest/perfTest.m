clear;
clc;
import matlab.unittest.TestSuite;

numTestRuns = 5;
filename = 'runtimes_<name_of_commit>';

runtimes = zeros(numTestRuns,1);
for k = 1:numTestRuns
%     suite = matlab.unittest.TestSuite.fromMethod(?performanceTest, 'performanceTestMIMO_01');
    suite = matlab.unittest.TestSuite.fromMethod(?performanceTest, 'performanceTestWheelBehavior_01');
    result = run(suite);
    runtimes(k) = result.Duration;
end
% save runtimes to disk
save(filename,'runtimes');