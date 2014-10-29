classdef gradientTest < matlab.unittest.TestCase
    %GRADIENTTEST Summary of this class goes here
    %   Detailed explanation goes here
        
    methods(Test)       
        function trainSingleValue(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % --------------------------------------
            % init network, 1-2-1 nn-framework
            % --------------------------------------
            p = 1;
            target = 1 + sin((pi/4)*p);

            net = nnfw.FeedForward(1, 2, 1);
            net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
            net.layers{1}.size = 2;
            net.IW{1} = [-0.27; -0.41];
            net.LW{2,1} = [0.09 -0.17];
            net.b{1,1} = [-0.48; -0.13];
            net.b{2,1} = [0.48];

            [~, grad] = net.train(p, target);
            
            % --------------------------------------
            % prepare starting weights in numeric calculation
            % --------------------------------------
            weights = net.getWeightVector();
            w111 = weights(1); %x // w111
            w211 = weights(2); %y // w211
            b11 = weights(3); %z // b11
            b21 = weights(4); %q // b21
            w112 = weights(5); %r // w112
            w122 = weights(6); %s // w122
            b12 = weights(7); %t // b12
            % --------------------------------------
            % prepare gradient matrix
            % --------------------------------------

            grad_derivated = zeros(length(p), net.getNumWeights());

            pIn = p;
            Q = length(p);
            for q = 1:Q
                p = pIn(q);
                grad_derivated(q, 1) = (2*p*w112*exp(- b11 - p*w111)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b11 - p*w111) + 1)^2;
                grad_derivated(q, 2) = (2*p*w122*exp(- b21 - p*w211)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b21 - p*w211) + 1)^2;
                grad_derivated(q, 3) = (2*w112*exp(- b11 - p*w111)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b11 - p*w111) + 1)^2;
                grad_derivated(q, 4) = (2*w122*exp(- b21 - p*w211)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b21 - p*w211) + 1)^2;

                grad_derivated(q, 5) = (2*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b11 - p*w111) + 1);
                grad_derivated(q, 6) = (2*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b21 - p*w211) + 1);
                grad_derivated(q, 7) = 2*b12 - 2*sin((pi*p)/4) + (2*w122)/(exp(- b21 - p*w211) + 1) + (2*w112)/(exp(- b11 - p*w111) + 1) - 2;
            end
            grad_derivated = sum(grad_derivated, 1);
            
            tc.assertThat(grad, IsEqualTo(grad_derivated, 'Within', AbsoluteTolerance(1e-22)));
        end
        
        function trainQValues(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % --------------------------------------
            % init network, 1-2-1 nn-framework
            % --------------------------------------
            p = (-2:0.1:2);
            target = 1 + sin((pi/4)*p);

            net = nnfw.FeedForward(1, 2, 1);
            net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
            net.layers{1}.size = 2;
            net.IW{1} = [-0.27; -0.41];
            net.LW{2,1} = [0.09 -0.17];
            net.b{1,1} = [-0.48; -0.13];
            net.b{2,1} = [0.48];

            [~, grad] = net.train(p, target);
            
            % --------------------------------------
            % prepare starting weights in numeric calculation
            % --------------------------------------
            weights = net.getWeightVector();
            w111 = weights(1); %x // w111
            w211 = weights(2); %y // w211
            b11 = weights(3); %z // b11
            b21 = weights(4); %q // b21
            w112 = weights(5); %r // w112
            w122 = weights(6); %s // w122
            b12 = weights(7); %t // b12
            % --------------------------------------
            % prepare gradient matrix
            % --------------------------------------

            grad_derivated = zeros(length(p), net.getNumWeights());

            pIn = p;
            Q = length(p);
            for q = 1:Q
                p = pIn(q);
                grad_derivated(q, 1) = (2*p*w112*exp(- b11 - p*w111)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b11 - p*w111) + 1)^2;
                grad_derivated(q, 2) = (2*p*w122*exp(- b21 - p*w211)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b21 - p*w211) + 1)^2;
                grad_derivated(q, 3) = (2*w112*exp(- b11 - p*w111)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b11 - p*w111) + 1)^2;
                grad_derivated(q, 4) = (2*w122*exp(- b21 - p*w211)*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b21 - p*w211) + 1)^2;

                grad_derivated(q, 5) = (2*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b11 - p*w111) + 1);
                grad_derivated(q, 6) = (2*(b12 - sin((pi*p)/4) + w122/(exp(- b21 - p*w211) + 1) + w112/(exp(- b11 - p*w111) + 1) - 1))/(exp(- b21 - p*w211) + 1);
                grad_derivated(q, 7) = 2*b12 - 2*sin((pi*p)/4) + (2*w122)/(exp(- b21 - p*w211) + 1) + (2*w112)/(exp(- b11 - p*w111) + 1) - 2;
            end
            grad_derivated = sum(grad_derivated, 1);
            
            tc.assertThat(grad, IsEqualTo(grad_derivated, 'Within', AbsoluteTolerance(1e-14)));
        end

        function trainVectorValues(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;

            load bodyfat_dataset

            % --------------------------------------
            % init network, 13-2-1 nn-framework
            % --------------------------------------
            p = bodyfatInputs;
            target = bodyfatTargets;

            net = nnfw.FeedForward(1, 2, 1);
            net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
            net.layers{1}.size = 2;
            net.configure(p, target);
            weights = rand(net.getNumWeights(),1);
            net.setWeights(weights);
%             net.LW{2,1} = [5.01 5.01];
%             net.b{2,1} = 7.01;

            [~, grad] = net.train(p, target);
            
            % --------------------------------------
            % configure weights
            % --------------------------------------   
            weights = net.getWeightVector();
            w1011 = weights(1);
            w1021 = weights(2);
            w1031 = weights(3);
            w1041 = weights(4);
            w1051 = weights(5);
            w1061 = weights(6);
            w1071 = weights(7);
            w1081 = weights(8);
            w1091 = weights(9);
            w1101 = weights(10);
            w1111 = weights(11);
            w1121 = weights(12);
            w1131 = weights(13);

            w2011 = weights(14);
            w2021 = weights(15);
            w2031 = weights(16);
            w2041 = weights(17);
            w2051 = weights(18);
            w2061 = weights(19);
            w2071 = weights(20);
            w2081 = weights(21);
            w2091 = weights(22);
            w2101 = weights(23);
            w2111 = weights(24);
            w2121 = weights(25);
            w2131 = weights(26);

            b11 = weights(27);
            b21 = weights(28);

            w112 = weights(29);
            w122 = weights(30);
%             w112 = 5.01;
%             w122 = 5.01;
            b12 = weights(31); 
%             b12 = 7.01; 
            
            grad_derivated = zeros(length(p), net.getNumWeights());

            Q = length(p);
            pIn = p;
            tIn = target;
            for q = 1:Q
                target = tIn(q);
                p_1 = pIn(1,q);
                p_2 = pIn(2,q);
                p_3 = pIn(3,q);
                p_4 = pIn(4,q);
                p_5 = pIn(5,q);
                p_6 = pIn(6,q);
                p_7 = pIn(7,q);
                p_8 = pIn(8,q);
                p_9 = pIn(9,q);
                p_10 = pIn(10,q);
                p_11 = pIn(11,q);
                p_12 = pIn(12,q);
                p_13 = pIn(13,q);

                grad_derivated(q, 1) = (2*p_1*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 2) = (2*p_2*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 3) = (2*p_3*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 4) = (2*p_4*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 5) = (2*p_5*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 6) = (2*p_6*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 7) = (2*p_7*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 8) = (2*p_8*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 9) = (2*p_9*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 10) = (2*p_10*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 11) = (2*p_11*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 12) = (2*p_12*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 13) = (2*p_13*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;

                grad_derivated(q, 14) = (2*p_1*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 15) = (2*p_2*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 16) = (2*p_3*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 17) = (2*p_4*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 18) = (2*p_5*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 19) = (2*p_6*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 20) = (2*p_7*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 21) = (2*p_8*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 22) = (2*p_9*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 23) = (2*p_10*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 24) = (2*p_11*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 25) = (2*p_12*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;
                grad_derivated(q, 26) = (2*p_13*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;

                grad_derivated(q, 27) = (2*w112*exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1)^2;
                grad_derivated(q, 28) = (2*w122*exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131)*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)^2;

                grad_derivated(q, 29) = (2*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1);
                grad_derivated(q, 30) = (2*(b12 - target + w112/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + w122/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1)))/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1);
                grad_derivated(q, 31) = 2*b12 - 2*target + (2*w112)/(exp(- b11 - p_1*w1011 - p_2*w1021 - p_3*w1031 - p_4*w1041 - p_5*w1051 - p_6*w1061 - p_7*w1071 - p_8*w1081 - p_9*w1091 - p_10*w1101 - p_11*w1111 - p_12*w1121 - p_13*w1131) + 1) + (2*w122)/(exp(- b21 - p_1*w2011 - p_2*w2021 - p_3*w2031 - p_4*w2041 - p_5*w2051 - p_6*w2061 - p_7*w2071 - p_8*w2081 - p_9*w2091 - p_10*w2101 - p_11*w2111 - p_12*w2121 - p_13*w2131) + 1);

            end
            grad_derivated = sum(grad_derivated, 1);
            
            tc.assertThat(grad, IsEqualTo(grad_derivated, 'Within', AbsoluteTolerance(1e-18)));            
            
        end
    end
    
end

