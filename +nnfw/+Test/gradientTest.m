classdef gradientTest < matlab.unittest.TestCase
    %GRADIENTTEST Summary of this class goes here
    %   Detailed explanation goes here
        
    methods(Test)       
        function trainSingleValue(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % --------------------------------------
            % init nn-framework
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
            w111 = -0.27; %x // w111
            w211 = -0.41; %y // w211
            b11 = -0.48; %z // b11
            b21 = -0.13; %q // b21
            w112 = 0.09; %r // w112
            w122 = -0.17; %s // w122
            b12 = 0.48; %t // b12
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
            
            tc.assertThat(grad, IsEqualTo(grad_derivated, 'Within', AbsoluteTolerance(eps(grad_derivated))));
        end
        
        function trainQValues(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % --------------------------------------
            % init nn-framework
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
            w111 = -0.27; %x // w111
            w211 = -0.41; %y // w211
            b11 = -0.48; %z // b11
            b21 = -0.13; %q // b21
            w112 = 0.09; %r // w112
            w122 = -0.17; %s // w122
            b12 = 0.48; %t // b12
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
            
            tc.assertThat(grad, IsEqualTo(grad_derivated, 'Within', AbsoluteTolerance(1e1*eps(grad_derivated))));
        end

        function trainVectorValues(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;

            load bodyfat_dataset
            
            % --------------------------------------
            % init weights
            % --------------------------------------
            w1011 = -0.27;
            w1021 = 0.09;
            w1031 = -0.13;
            w1041 = -0.27;
            w1051 = 0.09;
            w1061 = -0.13;
            w1071 = -0.41;
            w1081 = -0.27;
            w1091 = 0.09;
            w1101 = -0.13;
            w1111 = -0.27;
            w1121 = 0.09;
            w1131 = -0.13;

            w2011 = -0.27;
            w2021 = 0.09;
            w2031 = -0.13;
            w2041 = -0.27;
            w2051 = 0.09;
            w2061 = -0.13;
            w2071 = -0.27;
            w2081 = 0.09;
            w2091 = -0.13;
            w2101 = -0.27;
            w2111 = 0.09;
            w2121 = -0.13;
            w2131 = -0.27;

            b11 = 0.09;
            b21 = -0.13;

            w112 = -0.27;
            w122 = 0.09;
            b12 = -0.13;            

            % --------------------------------------
            % init nn-framework
            % --------------------------------------
            p = bodyfatInputs;
            target = bodyfatTargets;

            IW = [w1011 w1021 w1031 w1041 w1051 w1061 w1071 w1081 w1091 w1101 w1111 w1121 w1131;
                w2011 w2021 w2031 w2041 w2051 w2061 w2071 w2081 w2091 w2101 w2111 w2121 w2131];
            LW = [w112 w122];
            b1 = [b11; b21];
            b2 = [b12];

            net = nnfw.FeedForward(1, 2, 1);
            net.layers{1}.f = nnfw.Util.Activation.LOGSIG;
            net.layers{1}.size = 2;
            net.IW{1} = IW;
            net.LW{2,1} = LW;
            net.b{1,1} = b1;
            net.b{2,1} = b2;

            [~, grad] = net.train(p, target);
            
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
