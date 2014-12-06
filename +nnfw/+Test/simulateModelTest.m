classdef simulateModelTest < matlab.unittest.TestCase
    %SIMULATEMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Test)
        function simulate(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % net inputs
            p = -2:.1:2;
            % target outputs
            target = [-0.997561736896123 -0.981447467801894 -0.950877751407839 -0.897133034312682 -0.813797521946112 -0.704493708077894 -0.581619237107517 -0.451706666131718 -0.310267418050693 -0.154823948903224 0.00536541052277945 0.158283113237866 0.303837981570416 0.448769699324840 0.586632000930743 0.704653873838973 0.805074089429702 0.891012294133800 0.953457143130934 0.987297732883102 0.996826647751369 0.985592312598012 0.951443976395130 0.890583252191210 0.806036078223126 0.705983865163401 0.590527943919085 0.454773952915796 0.305474636727351 0.156282796662757 0.00447383408630536 -0.160519854557070 -0.321826594987321 -0.461929187789909 -0.588934846680663 -0.707959889672132 -0.811918767832918 -0.892826049218841 -0.949365068440590 -0.985796794575064 -1.00802736731586];
            % --------------------------------------
            % init network, nn framework
            % --------------------------------------
            net = nnfw.FeedForward(2);
            net.configure(p, target);
            net.IW{1} = [4.79265243547722;2.87374463693854;5.45308948020087;3.61749952554459;4.07605335110360;-4.38424970883532;-4.42917231892635;-3.11689441072215;3.69111054954978;-3.80346312430950];
            net.LW{2,1} = [0.445555269425985 -0.379802619908990 -0.160384913949304 -0.347007891028324 -0.147386402457643 -0.143665432461282 -0.200762924163457 -0.448144004535699 0.229483724642114 -0.295794869162931];
            net.b{1,1} = [-15.1518691289605;-4.16215426299049;-5.97660483017683;-2.71632130192552;-1.46844023089440;-1.50949649719622;-2.97761917385354;-3.34264102648195;5.62903784799786;-15.5936580447774];
            net.b{2,1} = [-0.876454292687631];
            
            Q = length(target);
            for q = 1:Q
              tc.assertThat(net.simulate(p(q), false), IsEqualTo(target(q), 'Within', AbsoluteTolerance(1e4*eps(target(q)))));
            end
        end
        
        function simulateTest_02(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % --------------------------------------
            % init training values
            % --------------------------------------
            input = (-5:.1:5);
            target = cos(pi*input/2);

            % --------------------------------------
            % init nn-framework 1-10-1
            % --------------------------------------
            net = nnfw.FeedForward(10);
            net.configure(input,target);
            net.initWeights();
            % --------------------------------------
            % old simulation algorithm - before performance optimization
            % --------------------------------------
            applyValueMapping = false;
            netSize = net.numLayers;
            inputTransFcn = net.layers{1}.f.f;
            inputLW = net.IW{1}; 
            inputBias = net.b{1};
            outputTransFcn = net.outputs{netSize}.f.f;
            outputLW = net.LW{netSize,netSize-1};
            outputBias = net.b{netSize};
            Q = size(input,2);
            a_expected = cell(Q,net.numLayers);
            outputSize = net.outputs{net.numLayers}.size;
            y_expected = zeros(outputSize,Q);
            for q = 1:Q
                for layer = 1:netSize
                    if layer == 1 % input layer
                        p = input(:,q);
                        a_expected{q, layer} = inputTransFcn( inputLW*p + inputBias );
                    elseif layer == netSize % output layer
                        p = a_expected{q, layer-1};
                        a_expected{q, layer} = outputTransFcn( outputLW*p + outputBias );
                    else % hidden layer
                        LW = net.LW{layer,layer-1};
                        p = a_expected{q, layer-1};
                        a_expected{q, layer} = net.layers{layer}.f.f( LW*p + net.b{layer} );
                    end
                end
                if applyValueMapping
%                     y_expected(:,q) = nnfw.Util.minmaxMappingRevert(a_expected{q,netSize}, net.minmaxTargetSettings);
                else
                    y_expected(:,q) = a_expected{q,netSize};
                end
            end
            % --------------------------------------
            % check if values of the new algorithm are the same
            % --------------------------------------
            [y, a] = net.simulate(input, false);
            tc.assertThat(y, IsEqualTo(y_expected, 'Within', AbsoluteTolerance(1e-15)));
            tc.assertThat(a, IsEqualTo(a_expected, 'Within', AbsoluteTolerance(1e-14)));
        end
        
        function simulateTest_03(tc)
            import matlab.unittest.constraints.IsEqualTo;
            import matlab.unittest.constraints.AbsoluteTolerance;
            % --------------------------------------
            % init training values
            % --------------------------------------
            input = (-5:.1:5);
            target = cos(pi*input/2);

            % --------------------------------------
            % init nn-framework 1-3-5-1
            % --------------------------------------
            net = nnfw.FeedForward([3 5]);
            net.configure(input,target);
            net.initWeights();
            % --------------------------------------
            % old simulation algorithm - before performance optimization
            % --------------------------------------
            applyValueMapping = false;
            netSize = net.numLayers;
            inputTransFcn = net.layers{1}.f.f;
            inputLW = net.IW{1}; 
            inputBias = net.b{1};
            outputTransFcn = net.outputs{netSize}.f.f;
            outputLW = net.LW{netSize,netSize-1};
            outputBias = net.b{netSize};
            Q = size(input,2);
            a_expected = cell(Q,net.numLayers);
            outputSize = net.outputs{net.numLayers}.size;
            y_expected = zeros(outputSize,Q);
            for q = 1:Q
                for layer = 1:netSize
                    if layer == 1 % input layer
                        p = input(:,q);
                        a_expected{q, layer} = inputTransFcn( inputLW*p + inputBias );
                    elseif layer == netSize % output layer
                        p = a_expected{q, layer-1};
                        a_expected{q, layer} = outputTransFcn( outputLW*p + outputBias );
                    else % hidden layer
                        LW = net.LW{layer,layer-1};
                        p = a_expected{q, layer-1};
                        a_expected{q, layer} = net.layers{layer}.f.f( LW*p + net.b{layer} );
                    end
                end
                if applyValueMapping
%                     y_expected(:,q) = nnfw.Util.minmaxMappingRevert(a_expected{q,netSize}, net.minmaxTargetSettings);
                else
                    y_expected(:,q) = a_expected{q,netSize};
                end
            end
            % --------------------------------------
            % check if values of the new algorithm are the same
            % --------------------------------------
            [y, a] = net.simulate(input, false);
            tc.assertThat(y, IsEqualTo(y_expected, 'Within', AbsoluteTolerance(1e-15)));
            tc.assertThat(a, IsEqualTo(a_expected, 'Within', AbsoluteTolerance(1e-14)));
        end
    end
    
end

