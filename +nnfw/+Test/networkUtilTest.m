classdef networkUtilTest < matlab.unittest.TestCase
    %NETWORKUTILTEST Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Test)
        function getNumWeightsTest_01(tc)
            % --------------------------------------
            % init network, 1-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,2,1);
            net.inputs{1}.size = 1;
            net.layers{1}.size = 2;
            net.outputs{2}.size = 1;
            
            expected = 7;
            
            tc.assertEqual(net.getNumWeights(), expected);
        end
        
        function getNumWeightsTest_02(tc)
            % --------------------------------------
            % init network, 13-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,2,1);
            net.inputs{1}.size = 13;
            net.layers{1}.size = 2;
            net.outputs{2}.size = 1;
            
            expected = 31;
            
            tc.assertEqual(net.getNumWeights(), expected);
        end
        
        function getNumWeightsTest_03(tc)
            % --------------------------------------
            % init network, 1-3-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,3,1);
            net.inputs{1}.size = 1;
            net.layers{1}.size = 3;
            net.layers{2}.size = 2;
            net.outputs{3}.size = 1;
            
            expected = 17;
            
            tc.assertEqual(net.getNumWeights(), expected);
        end
        
        function getWeightVectorTest_01(tc)
            % --------------------------------------
            % init network, 1-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,2,1);
            net.inputs{1}.size = 1;
            net.layers{1}.size = 2;
            net.outputs{2}.size = 1;
            
            expected = rand(7,1);
            
            net.IW{1} = expected(1:2);
            net.b{1,1} = expected(3:4);
            net.LW{2,1} = expected(5:6)';
            net.b{2,1} = expected(7);
            
            tc.assertEqual(net.getWeightVector(), expected);
        end
        
        function getWeightVectorTest_02(tc)
            % --------------------------------------
            % init network, 13-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,2,1);
            net.inputs{1}.size = 13;
            net.layers{1}.size = 2;
            net.outputs{2}.size = 1;
            
            expected = rand(31,1);
            
            net.IW{1} = expected(1:26);
            net.b{1,1} = expected(27:28);
            net.LW{2,1} = expected(29:30)';
            net.b{2,1} = expected(31);
            
            tc.assertEqual(net.getWeightVector(), expected);
        end
        
        function getWeightVectorTest_03(tc)
            % --------------------------------------
            % init network, 1-3-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,3,1);
            net.inputs{1}.size = 1;
            net.layers{1}.size = 3;
            net.layers{2}.size = 2;
            net.outputs{3}.size = 1;
            
            expected = rand(17,1);
            
            net.IW{1} = expected(1:3);
            net.b{1,1} = expected(4:6);
            net.LW{2,1} = expected(7:12);
            net.b{2,1} = expected(13:14);
            net.LW{3,1} = expected(15:16)';
            net.b{3,1} = expected(17);

            tc.assertEqual(net.getWeightVector(), expected);
        end
    end
    
end

