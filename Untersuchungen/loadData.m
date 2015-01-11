%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%            Model Wheel Adaper Identification AK2MR FNN:             %%%
%%%                                                                     %%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
% clear all;
clear;

%% pathes to function-files, measuring data, identified models: 
% path(path,'.\functions');          
% path(path,'.\measurements');

%% data sets: (use xxx_RSP)
filename = '01_APRBS_APK1_sim1_DRV_100Pct_RSP.tim';

% rpc2mat:
[DATA,dt,header] = rpc2mat(filename);
Data = double(DATA.signals);


%% system constants:
scale2kilo = 1e-3;
Area_X = 1935.0;        % mm^2
Area_Brake = 2451.0;    % mm^2
Faktor_Fz = 3978;       % mm^2

%% extract signals from Data:
% force / torque:
Longitudinal_Kraft = Data(:,8); % check: DATA.channels
Lateral_Kraft = Data(:,19);
Vertikal_Kraft = Data(:,31);
Sturz_Moment = Data(:,43);
Bremse_Moment = Data(:,54);
Lenkung_Moment = Data(:,64);

% displacement / angles:
Longitudinal_Weg = Data(:,7); % check: DATA.channels
Lateral_Weg = Data(:,18);
Vertikal_Weg = Data(:,29);
Sturz_Winkel = Data(:,42);
Bremse_Winkel = Data(:,53);
Lenkung_Winkel = Data(:,63);

% actuator: KMD, DeltaP, LVDT
Longitudinal_DeltaP = Data(:,74); % check: DATA.channels
Lateral1_KMD = Data(:,78); 
Lateral2_KMD = Data(:,82);
Lateral3_KMD = Data(:,87); 
Vertikal_DeltaP = Data(:,91);
Bremse_DeltaP = Data(:,95);

Longitudinal_LVDT = Data(:,75); 
Lateral1_LVDT = Data(:,80);
Lateral2_LVDT = Data(:,84);
Lateral3_LVDT = Data(:,89);
Vertikal_LVDT = Data(:,92);
Bremse_LVDT = Data(:,96);

% store actuator signals:
FXAK = Longitudinal_Kraft;
FYAK = Lateral_Kraft;
FZAK = Vertikal_Kraft;
MXAK = Sturz_Moment;
MYAK = Bremse_Moment;
MZAK = Lenkung_Moment;

SXAK = Longitudinal_Weg;
SYAK = Lateral_Weg;
SZAK = Vertikal_Weg;
AXAK = Sturz_Winkel;
AYAK = Bremse_Winkel;
AZAK = Lenkung_Winkel;

% measuring wheel:
FXMR = Data(:,1);
FYMR = Data(:,2);
FZMR = Data(:,3);
MXMR = Data(:,4);
MYMR = Data(:,5);
MZMR = Data(:,6);

%% actuator forces:
FX_actuator = 0.1 * Longitudinal_DeltaP * Area_X * scale2kilo;
FZ_actuator = 0.1 * Vertikal_DeltaP * Faktor_Fz * scale2kilo;
MY_actuator = 0.1 * Bremse_DeltaP * Area_Brake * scale2kilo;
LAT1_actuator = Lateral1_KMD;
LAT2_actuator = Lateral2_KMD;
LAT3_actuator = Lateral3_KMD;