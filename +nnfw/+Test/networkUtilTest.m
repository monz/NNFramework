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
        
        function setWeights_01(tc)
            % --------------------------------------
            % init network, 1-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,2,1);
            net.inputs{1}.size = 1;
            net.layers{1}.size = 2;
            net.outputs{2}.size = 1;
            
            weights = rand(7,1);
            net.setWeights(weights);
            
            tc.assertEqual(net.IW{1}, weights(1:2,1));
            tc.assertEqual(net.b{1}, weights(3:4,1));
            tc.assertEqual(net.LW{2}, weights(5:6,1)');
            tc.assertEqual(net.b{2}, weights(7:7,1));
        end
        
        function setWeights_02(tc)
            % --------------------------------------
            % init network, 13-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,2,1);
            net.inputs{1}.size = 13;
            net.layers{1}.size = 2;
            net.outputs{2}.size = 1;
            
            weights = rand(31,1);
            net.setWeights(weights);

            % assert LW{1} dimensions first
            R = net.inputs{1}.size;
            S_1 = net.layers{1}.size;
            tc.assertEqual(size(net.IW{1}), [S_1 R]);
            % assert LW{1} weights
            tc.assertEqual(net.IW{1}(1,:), weights(1:13,1)');
            tc.assertEqual(net.IW{1}(2,:), weights(14:26,1)');
            tc.assertEqual(net.b{1}, weights(27:28,1));
            
            % assert LW{2} dimensions first
            S_1 = net.layers{1}.size;
            S_2 = net.outputs{2}.size;
            tc.assertEqual(size(net.LW{2}), [S_2 S_1]);
            % assert LW{2} weights
            tc.assertEqual(net.LW{2}, weights(29:30,1)');
            tc.assertEqual(net.b{2}, weights(31));
        end
        
        function setWeights_03(tc)
            % --------------------------------------
            % init network, 1-3-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(1,3,1);
            net.inputs{1}.size = 1;
            net.layers{1}.size = 3;
            net.layers{2}.size = 2;
            net.outputs{3}.size = 1;
            
            weights = rand(17,1);
            net.setWeights(weights);
            
            % assert LW{1} dimensions first
            R = net.inputs{1}.size;
            S_1 = net.layers{1}.size;
            tc.assertEqual(size(net.IW{1}), [S_1 R]);
            % assert LW{1} weights
            tc.assertEqual(net.IW{1}, weights(1:3,1));
            tc.assertEqual(net.b{1}, weights(4:6,1));
            
            % assert LW{2} dimensions first
            S_1 = net.layers{1}.size;
            S_2 = net.layers{2}.size;
            tc.assertEqual(size(net.LW{2}), [S_2 S_1]);
            % assert LW{2} weights
            tc.assertEqual(net.LW{2}(1,:), weights(7:9,1)');
            tc.assertEqual(net.LW{2}(2,:), weights(10:12,1)');
            tc.assertEqual(net.b{2}, weights(13:14));
            
            % assert LW{3} dimensions first
            S_2 = net.layers{2}.size;
            S_3 = net.outputs{3}.size;
            tc.assertEqual(size(net.LW{3}), [S_3 S_2]);
            % assert LW{3} weights
            tc.assertEqual(net.LW{3}, weights(15:16,1)');
            tc.assertEqual(net.b{3}, weights(17));
        end
    end
    
end

