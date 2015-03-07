clear;
clc;

maxTime = 0;
% testFiles = {
%     'runtimes_extract_simple_offset_variable', ...
%     'runtimes_improved_calculation_of_simulation', ...
%     'runtimes_implemented_a_fast_version_of_MSE_error_function', ...
%     'runtimes_some_minor_improvements', ...
%     'runtimes_extracted_and_splitted_sensitivity_computation_in_own_loop', ...
%     'runtimes_calculate_row_index_only_if_necessary', ...
%     'runtimes_added_own_simple_implementation_of_num2cell', ...
%     'runtimes_improved_performance_of_error_plot_function', ...
%     'runtimes_clean_up_temporary_allocated_memory'
% };

testFiles = {
    'runtimes_69aca30', ...  % 1)
    'runtimes_45b1dc1', ...  % 2)
    'runtimes_2d79ba0', ...  % 3)
    'runtimes_1aa2cb6', ...  % 4)
    'runtimes_5113fd2', ...  %5)
    'runtimes_0fae67a', ...  % 6)
    'runtimes_8b0d5e0', ...  % 7)
    'runtimes_973239e', ...  % 8)
    'runtimes_8be6986', ...  % 9)
    'runtimes_6389b2e'  % 10)
};
numTestRuns = length(testFiles);
testTimes = cell(numTestRuns,2);
for k = 1:numTestRuns
    load(testFiles{k});
    testTimes{k} = runtimes;
    testTimes{k,2} = mean(runtimes);
    maxTime = max(maxTime, max(runtimes));
end

figure();
grid on;
hold on
title('Performance Tuning Test Runs; Function Fitting');
% title('Performance Tuning Test Runs; Parameter Identification');
xlabel('Tuning Process Cycles');
ylabel('Time [s]');
axis([1 numTestRuns 0 floor(maxTime)+5])
set(gca, 'XTick', 1:numTestRuns);
for k = 1:numTestRuns
    ts = plot(k, testTimes{k,1}, 'gx', 'linewidth', 2); % test samples
    mv = plot(k, testTimes{k,2}, 'rx', 'linewidth', 2); % mean values
end
plot(1:numTestRuns, cell2mat(testTimes(:,2)), '-k', 'linewidth', 1); % connect mean values
legend([ts(1), mv(1)], 'Test Run Execution Time', 'Test Run Execution Time Mean')
hold off