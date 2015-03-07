clear;
clc;
import matlab.unittest.TestSuite;

numTestRuns = 5;
% filename = 'runtimes_<name_of_commit>';

% filename = 'runtimes_69aca30';    % 1)
% filename = 'runtimes_45b1dc1';   % 2)
% filename = 'runtimes_2d79ba0'; % 3)
% filename = 'runtimes_1aa2cb6';  % 4)
% filename = 'runtimes_5113fd2'; %5)
% filename = 'runtimes_0fae67a';   % 6)
% filename = 'runtimes_8b0d5e0';   % 7)
% filename = 'runtimes_973239e';   % 8)
% filename = 'runtimes_8be6986';   % 9)
filename = 'runtimes_6389b2e';   % 10)

runtimes = zeros(numTestRuns,1);
for k = 1:numTestRuns
%     suite = matlab.unittest.TestSuite.fromMethod(?performanceTest, 'performanceTestMIMO_01');
    suite = matlab.unittest.TestSuite.fromMethod(?performanceTest, 'performanceTestWheelBehavior_01');
    result = run(suite);
    runtimes(k) = result.Duration;
end
% save runtimes to disk
save(filename,'runtimes');