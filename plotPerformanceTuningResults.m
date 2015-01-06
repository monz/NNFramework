clear;
clc;

maxTime = 0;
testFiles = {
    'runtimes_no_improvements', ...
    'runtimes_improved_calculation_of_simulation', ...
    'runtimes_implemented_fast_MSE', ...
    'runtimes_some_minor_improvements', ...
    'runtimes_extracted_and_splitted_in_own_loop', ...
    'runtimes_calculate_row_index_only_if_necessary', ...
    'runtimes_added_own_num2cell', ...
    'runtimes_improved_performance_of_error_plot_function', ...
    'runtimes_clean_up_allocated_memory'
};
numTestRuns = length(testFiles);
testTimes = cell(numTestRuns,2);
for k = 1:numTestRuns
    load(testFiles{k});
    testTimes{k} = runtimes;
    testTimes{k,2} = mean(runtimes);
    maxTime = max(maxTime, max(runtimes));
end

hold on
title('Performance Tuning Test Runs MIMO');
xlabel('Tuning Iteration');
ylabel('Time in Seconds');
axis([1 numTestRuns 0 floor(maxTime)+5])
set(gca, 'XTick', 1:numTestRuns);
for k = 1:numTestRuns
    plot(k, testTimes{k,1}, 'gx', 'linewidth', 2); % test samples
    plot(k, testTimes{k,2}, 'rx', 'linewidth', 3); % mean values
end
plot(1:numTestRuns, cell2mat(testTimes(:,2)), '-k', 'linewidth', 2); % connect mean values
hold off