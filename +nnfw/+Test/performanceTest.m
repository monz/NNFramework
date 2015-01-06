classdef performanceTest < matlab.unittest.TestCase
    %PERFORMANCETEST Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Test)
        function performanceTestMIMO_01(tc)
            close all;
            % --------------------------------------
            % init training values
            % --------------------------------------
            p = [-2:.1:2; -2:.1:2];
            t = [sin(pi*p(1,:)/2); cos(pi*p(2,:)/2)];

            % --------------------------------------
            % init nn-framework
            % --------------------------------------
            net = nnfw.FeedForward(10);
            net.configure(p,t);
            net.optim.abortThreshold = 1e-10;
            net.optim.maxIter = 100;

            % --------------------------------------
            % train network
            % --------------------------------------
            profile on
            [E, ~, output, lambda, jacobian] = net.train(p,t);
            profile off
            profInfo = profile('info');
        end
    end
    
end

