classdef performanceTest < matlab.unittest.TestCase
    %PERFORMANCETEST build up tests to measure the performance with fixed circumstances.
    
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
            [E, ~, output, jacobian] = net.train(p,t);
            profile off
            profInfo = profile('info');
            % TODO: save profile to disk
        end
        
        function performanceTestWheelBehavior_01(tc)
            close all;
            clear;

            %% data sets: (use xxx_RSP)
            filename = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';

            % rpc2mat:
            [DATA,dt,header] = rpc2mat(filename);
            Data = double(DATA.signals);

            % actuator: KMD, DeltaP, LVDT
            Longitudinal_DeltaP = Data(:,74); % check: DATA.channels
            Bremse_DeltaP = Data(:,95);
            Bremse_LVDT = Data(:,96);

            % measuring wheel:
            FXMR = Data(:,1);
            MYMR = Data(:,5);

            %% identification:
            x = [FXMR';MYMR'];
            t = [Longitudinal_DeltaP';Bremse_DeltaP';Bremse_LVDT'];

            net = nnfw.FeedForward(10);
            net.configure(x,t);
            net.optim.maxIter = 3;
            profile on
            net.train(x,t);
            profile off
            profInfo = profile('info');
            % TODO: save profile to disk
        end
    end
    
end

