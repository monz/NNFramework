classdef mseTest < matlab.unittest.TestCase
    %MSETEST validates the MSE-Fast implementation.
    %   The former error computation of the neural network is compared to
    %   the new fast MSE implementation. The results have to be the same,
    %   obviously.
    
    methods (Test)
        function mseTest_01(tc)
            % --------------------------------------
            % init training values
            % --------------------------------------
            p = (-5:.1:5);
            t = cos(pi*p/2);
            
            % --------------------------------------
            % init nn-framework 1-5-1 network
            % --------------------------------------
            net = nnfw.FeedForward(5);
            net.configure(p,t);
            net.initWeights();
            % simulate network with initial random weights - no training
            % needed here
            y_d = net.simulate(p);
            % --------------------------------------
            % calculate E(w) with normal MSE function
            % --------------------------------------            
            Q = size(p, 2);
            ENormal = 0;
            for q = 1:Q
                ENormal = ENormal + nnfw.Util.mse(y_d(:, q), t(:, q));
            end
            % --------------------------------------
            % calculate E(w) with performant version of MSE
            % --------------------------------------
            EFast = nnfw.Util.mseFast(y_d, t);
            % --------------------------------------
            % check if results are the same
            % --------------------------------------
            tc.assertEqual(EFast, ENormal);
        end
        
        function mseTest_02(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % --------------------------------------
            % init training values
            % --------------------------------------
            p = [-2:.1:2; -2:.1:2];
            t = [sin(pi*p(1,:)/2); cos(pi*p(2,:)/2)];
            
            % --------------------------------------
            % init nn-framework 2-10-2 network
            % --------------------------------------
            net = nnfw.FeedForward(10);
            net.configure(p,t);
            net.initWeights();
            % simulate network with initial random weights - no training
            % needed here
            y_d = net.simulate(p);
            % --------------------------------------
            % calculate E(w) with normal MSE function
            % --------------------------------------            
            Q = size(p, 2);
            ENormal = 0;
            for q = 1:Q
                ENormal = ENormal + nnfw.Util.mse(y_d(:, q), t(:, q));
            end
            % --------------------------------------
            % calculate E(w) with performant version of MSE
            % --------------------------------------
            EFast = nnfw.Util.mseFast(y_d, t);
            % --------------------------------------
            % check if results are the same
            % --------------------------------------
            tc.assertThat(EFast, IsEqualTo(ENormal, 'Within', AbsoluteTolerance(1e-12)));
        end
    end
    
end

