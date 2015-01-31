classdef networkUtilTest < matlab.unittest.TestCase
    %NETWORKUTILTEST validates a collection of useful network utilization functions.
    
    methods(Test)
        function getNumWeightsTest_01(tc)
            % --------------------------------------
            % init network, 1-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 1;
            net.outputs{2}.size = 1;
            
            expected = 7;
            
            tc.assertEqual(net.getNumWeights(), expected);
        end
        
        function getNumWeightsTest_02(tc)
            % --------------------------------------
            % init network, 13-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 13;
            net.outputs{2}.size = 1;
            
            expected = 31;
            
            tc.assertEqual(net.getNumWeights(), expected);
        end
        
        function getNumWeightsTest_03(tc)
            % --------------------------------------
            % init network, 1-3-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward([3, 2]);
            net.inputs{1}.size = 1;
            net.outputs{3}.size = 1;
            
            expected = 17;
            
            tc.assertEqual(net.getNumWeights(), expected);
        end
        
        function getWeightVectorTest_01(tc)
            % --------------------------------------
            % init network, 1-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 1;
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
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 13;
            net.outputs{2}.size = 1;
            
            expected = rand(31,1);
            
            net.IW{1} = zeros(2, 13);
            net.IW{1}(1,:) = expected(1:13);
            net.IW{1}(2,:) = expected(14:26);
            net.b{1,1} = expected(27:28);
            net.LW{2,1} = zeros(1, 2);
            net.LW{2,1}(1,:) = expected(29:30);
            net.b{2,1} = expected(31);
            
            tc.assertEqual(net.getWeightVector(), expected);
        end
        
        function getWeightVectorTest_03(tc)
            % --------------------------------------
            % init network, 1-3-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward([3, 2]);
            net.inputs{1}.size = 1;
            net.outputs{3}.size = 1;
            
            expected = rand(17,1);
            
            net.IW{1} = zeros(3, 1);
            net.IW{1}(:,1) = expected(1:3)';
            net.b{1,1} = expected(4:6);
            
            net.LW{2,1} = zeros(2, 3);
            net.LW{2,1}(1,:) = expected(7:9);
            net.LW{2,1}(2,:) = expected(10:12);
            net.b{2,1} = expected(13:14);
            
            net.LW{3,2} = zeros(1, 2);
            net.LW{3,2}(1,:) = expected(15:16);
            net.b{3,1} = expected(17);

            tc.assertEqual(net.getWeightVector(), expected);
        end
        
        function getWeightVectorTest_04(tc)
            % --------------------------------------
            % init network, 1-3-2-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward([3, 2, 2]);
            net.inputs{1}.size = 1;
            net.outputs{4}.size = 1;
            
            expected = rand(23,1);
            
            net.IW{1} = zeros(3, 1);
            net.IW{1}(:, 1) = expected(1:3)';
            net.b{1,1} = expected(4:6);
            
            net.LW{2,1} = zeros(2, 3);
            net.LW{2,1}(1,:) = expected(7:9);
            net.LW{2,1}(2,:) = expected(10:12);
            net.b{2,1} = expected(13:14);
            
            net.LW{3,2} = zeros(2);
            net.LW{3,2}(1,:) = expected(15:16);
            net.LW{3,2}(2,:) = expected(17:18);
            net.b{3,1} = expected(19:20);
            
            net.LW{4,3} = zeros(1, 2);
            net.LW{4,3}(1,:) = expected(21:22);
            net.b{4,1} = expected(23);

            tc.assertEqual(net.getWeightVector(), expected);
        end
        
        function getWeightVectorTest_05(tc)
            % --------------------------------------
            % init network, 2-2-2 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 2;
            net.outputs{2}.size = 2;
            
            expected = rand(12,1);
            % fill neural network weights
            net.IW{1} = zeros(2);
            net.IW{1}(1,:) = expected(1:2);
            net.IW{1}(2,:) = expected(3:4);
            net.b{1,1} = expected(5:6);
            net.LW{2,1} = zeros(2);
            net.LW{2,1}(1,:) = expected(7:8);
            net.LW{2,1}(2,:) = expected(9:10);
            net.b{2,1} = expected(11:12);

            tc.assertEqual(net.getWeightVector(), expected);
        end
        
        function setWeights_01(tc)
            % --------------------------------------
            % init network, 1-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 1;
            net.outputs{2}.size = 1;
            
            weights = rand(7,1);
            net.setWeights(weights);
            
            tc.assertEqual(net.IW{1}, weights(1:2,1));
            tc.assertEqual(net.b{1}, weights(3:4,1));
            tc.assertEqual(net.LW{2,1}, weights(5:6,1)');
            tc.assertEqual(net.b{2}, weights(7:7,1));
        end
        
        function setWeights_02(tc)
            % --------------------------------------
            % init network, 13-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 13;
            net.outputs{2}.size = 1;
            
            weights = rand(31,1);
            net.setWeights(weights);

            % assert LW{1} dimensions first
            R = net.inputs{1}.size;
            S_1 = net.layers{1}.size;
            tc.assertEqual(size(net.IW{1}), [S_1 R]);
            % assert LW{1} weights
            tc.assertEqual(net.IW{1}(1, :), weights(1:13,1)'); % check row 1
            tc.assertEqual(net.IW{1}(2, :), weights(14:26,1)'); % check row 2
            tc.assertEqual(net.b{1}, weights(27:28,1));
            
            % assert LW{2} dimensions first
            S_1 = net.layers{1}.size;
            S_2 = net.outputs{2}.size;
            tc.assertEqual(size(net.LW{2,1}), [S_2 S_1]);
            % assert LW{2} weights
            tc.assertEqual(net.LW{2,1}, weights(29:30,1)');
            tc.assertEqual(net.b{2}, weights(31));
        end
        
        function setWeights_03(tc)
            % --------------------------------------
            % init network, 1-3-2-1 nn framework
            % --------------------------------------
            net = nnfw.FeedForward([3, 2]);
            net.inputs{1}.size = 1;
            net.outputs{3}.size = 1;
            
            weights = rand(17,1);
            net.setWeights(weights);
            
            % assert LW{1} dimensions first
            R = net.inputs{1}.size;
            S_1 = net.layers{1}.size;
            tc.assertEqual(size(net.IW{1}), [S_1 R]);
            % assert LW{1} weights
            tc.assertEqual(net.IW{1}(1, :), weights(1,1)); % check row 1
            tc.assertEqual(net.IW{1}(2, :), weights(2,1)); % check row 2
            tc.assertEqual(net.IW{1}(3, :), weights(3,1)); % check row 3
            tc.assertEqual(net.b{1}, weights(4:6,1));
            
            % assert LW{2} dimensions first
            S_1 = net.layers{1}.size;
            S_2 = net.layers{2}.size;
            tc.assertEqual(size(net.LW{2,1}), [S_2 S_1]);
            % assert LW{2} weights
            tc.assertEqual(net.LW{2,1}(1, :), weights(7:9,1)'); % check row 1
            tc.assertEqual(net.LW{2,1}(2, :), weights(10:12,1)'); % check row 2
            tc.assertEqual(net.b{2}, weights(13:14));
            
            % assert LW{3} dimensions first
            S_2 = net.layers{2}.size;
            S_3 = net.outputs{3}.size;
            tc.assertEqual(size(net.LW{3,2}), [S_3 S_2]);
            % assert LW{3} weights
            tc.assertEqual(net.LW{3,2}(1, :), weights(15:16,1)'); % check row 1
            tc.assertEqual(net.b{3}, weights(17));
        end
        
        function setWeights_04(tc)
            % --------------------------------------
            % init network, 2-2-2 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 2;
            net.outputs{2}.size = 2;
            
            weights = rand(12,1);
            net.setWeights(weights);
            
            % assert LW{1} dimensions first
            R = net.inputs{1}.size;
            S_1 = net.layers{1}.size;
            tc.assertEqual(size(net.IW{1}), [S_1 R]);
            % assert LW{1} weights
            tc.assertEqual(net.IW{1}(1, :), weights(1:2,1)'); % check row 1
            tc.assertEqual(net.IW{1}(2, :), weights(3:4,1)'); % check row 2
            tc.assertEqual(net.b{1}, weights(5:6,1));
            
            % assert LW{2} dimensions first
            S_1 = net.layers{1}.size;
            S_2 = net.outputs{2}.size;
            tc.assertEqual(size(net.LW{2,1}), [S_2 S_1]);
            % assert LW{2} weights
            tc.assertEqual(net.LW{2,1}(1, :), weights(7:8,1)'); % check row 1
            tc.assertEqual(net.LW{2,1}(2, :), weights(9:10,1)'); % check row 2
            tc.assertEqual(net.b{2}, weights(11:12,1));
        end
        
        function setWeights_05(tc)
            % --------------------------------------
            % init network, 2-2-4 nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.inputs{1}.size = 2;
            net.outputs{2}.size = 4;
            
            weights = rand(18,1);
            net.setWeights(weights);
            
            % assert LW{1} dimensions first
            R = net.inputs{1}.size;
            S_1 = net.layers{1}.size;
            tc.assertEqual(size(net.IW{1}), [S_1 R]);
            % assert weights
            tc.assertEqual(net.IW{1}(1, :), weights(1:2,1)'); % check row 1
            tc.assertEqual(net.IW{1}(2, :), weights(3:4,1)'); % check row 2
            tc.assertEqual(net.b{1}, weights(5:6,1));
            
            % assert LW{2} dimensions first
            S_1 = net.layers{1}.size;
            S_2 = net.outputs{2}.size;
            tc.assertEqual(size(net.LW{2,1}), [S_2 S_1]);
            % assert LW{2} weights
            tc.assertEqual(net.LW{2,1}(1, :), weights(7:8,1)'); % check row 1
            tc.assertEqual(net.LW{2,1}(2, :), weights(9:10,1)'); % check row 2
            tc.assertEqual(net.LW{2,1}(3, :), weights(11:12,1)'); % check row 3
            tc.assertEqual(net.LW{2,1}(4, :), weights(13:14,1)'); % check row 4
            tc.assertEqual(net.b{2}, weights(15:18,1));
        end
    end
    
end

