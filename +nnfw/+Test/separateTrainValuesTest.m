classdef separateTrainValuesTest < matlab.unittest.TestCase
    %SEPARATETRAINVALUESTEST Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Test)
        function separateTrainValuesTest_01(tc)
            input = [1 2 3 4 5 6 7 8 9 0];
            target = [9 9 2 1 6 2 7 8 4 1];
            
            values = nnfw.Util.separateTrainingValues(input, target, 0.20, 0.05);
            
            % check if all values have been set
            tc.assertFalse(ismember(NaN,values{1,1}));
            tc.assertFalse(ismember(NaN,values{1,2}));
            tc.assertFalse(ismember(NaN,values{2,1}));
            tc.assertFalse(ismember(NaN,values{2,2}));
            tc.assertFalse(ismember(NaN,values{3,1}));
            tc.assertFalse(ismember(NaN,values{3,2}));
        end
        
        function separateTrainValuesTest_02(tc)
            input = [1 2 3 4 5 6 7 8 9 0];
            target = [9 9 2 1 6 2 7 8 4 1];
            
            values = nnfw.Util.separateTrainingValues(input, target, 0.20, 0.05);
            
            expectedNumTrainElems = 7;
            expectedNumValidateElems = 2;
            expectedNumTestElems = 1;
            
            % check if input and target size are of expected size and equal
            tc.assertEqual(length(values{1,1}), expectedNumTrainElems);
            tc.assertEqual(length(values{1,2}), expectedNumTrainElems);
            tc.assertEqual(length(values{2,1}), expectedNumValidateElems);
            tc.assertEqual(length(values{2,2}), expectedNumValidateElems);
            tc.assertEqual(length(values{3,1}), expectedNumTestElems);
            tc.assertEqual(length(values{3,2}), expectedNumTestElems);
        end
        
        function separateTrainValuesTest_03(tc)
            input = rand(1, 100);
            target = rand(1, 100);
            
            values = nnfw.Util.separateTrainingValues(input, target, 0.20, 0.05);
            
            expectedNumTrainElems = 75;
            expectedNumValidateElems = 20;
            expectedNumTestElems = 5;
            
            % check if input and target size are of expected size and equal
            tc.assertEqual(length(values{1,1}), expectedNumTrainElems);
            tc.assertEqual(length(values{1,2}), expectedNumTrainElems);
            tc.assertEqual(length(values{2,1}), expectedNumValidateElems);
            tc.assertEqual(length(values{2,2}), expectedNumValidateElems);
            tc.assertEqual(length(values{3,1}), expectedNumTestElems);
            tc.assertEqual(length(values{3,2}), expectedNumTestElems);
        end
        
        function separateTrainValuesTest_04(tc)
            input = rand(1, 33);
            target = rand(1, 33);
            
            values = nnfw.Util.separateTrainingValues(input, target, 0.20, 0.05);
            
            expectedNumTrainElems = 25;
            expectedNumValidateElems = 6;
            expectedNumTestElems = 2;
            
            % check if input and target size are of expected size and equal
            tc.assertEqual(length(values{1,1}), expectedNumTrainElems);
            tc.assertEqual(length(values{1,2}), expectedNumTrainElems);
            tc.assertEqual(length(values{2,1}), expectedNumValidateElems);
            tc.assertEqual(length(values{2,2}), expectedNumValidateElems);
            tc.assertEqual(length(values{3,1}), expectedNumTestElems);
            tc.assertEqual(length(values{3,2}), expectedNumTestElems);
        end
    end
    
end

