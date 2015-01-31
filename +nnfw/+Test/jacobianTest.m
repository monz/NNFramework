classdef jacobianTest < matlab.unittest.TestCase
    %JACOBIANTEST validates the computation of the jacobian matrix.
    %   The examples results have been evaluated by hand and are compared
    %   to the implemented jacobian algorithm.
    
    methods(Test)
        function jacobianSimpleTest_01(tc)
            p = [1 2];
            t = [1 2];
            
            % --------------------------------------
            % init nn-framework simple jacobian test
            % --------------------------------------
            net = nnfw.FeedForward(1);
            net.configure(p,t);
            net.layers{1}.f = nnfw.Util.Activation.QUAD;
            weights = [1 0 2 1]';
            net.setWeights(weights);
            
            costFcn = net.makeCostFcn2(@nnfw.Util.mse, p, t);
            [~, jacobian] = costFcn(net.getWeightVector());
           
            expected = [-4 -4 -1 -1; -16 -8 -4 -1];
            
            tc.assertEqual(jacobian, expected);
        end
        
        function jacobianSimpleTest_02(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;

            p = [1 2];
            t = [1 2];
            
            % --------------------------------------
            % init nn-framework simple jacobian test
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.configure(p,t);
            weights = [1 2 1 0 0.5 1 0]';
            net.setWeights(weights);
            
            costFcn = net.makeCostFcn2(@nnfw.Util.mse, p, t);
            [~, jacobian] = costFcn(net.getWeightVector());
            expected = [-0.0353254124 -0.07065082481 -0.0353254124 -0.07065082481 -0.9640275801 -0.9640275801 -1;
                        -0.009866037139 -0.002681905122 -0.00493301857 -0.001340950761 -0.9950547537 -0.9993292997 -1];
            
            tc.assertThat(jacobian, IsEqualTo(expected, 'Within', AbsoluteTolerance(1e-8)));
        end
    end
end
