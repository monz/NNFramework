function outdata = loadDataF(filename)
%LOADDATAF Loads test data from data structure
%  
%   filename:   name of the file whose data should get loaded

    % rpc2mat:
    [DATA,dt,header] = rpc2mat(filename);
    Data = double(DATA.signals);

    %% system constants:
    outdata.scale2kilo = 1e-3;
    outdata.Area_X = 1935.0;        % mm^2
    outdata.Area_Brake = 2451.0;    % mm^2
    outdata.Faktor_Fz = 3978;       % mm^2

    %% extract signals from Data:
    % force / torque:
    outdata.Longitudinal_Kraft = Data(:,8); % check: DATA.channels
    outdata.Lateral_Kraft = Data(:,19);
    outdata.Vertikal_Kraft = Data(:,31);
    outdata.Sturz_Moment = Data(:,43);
    outdata.Bremse_Moment = Data(:,54);
    outdata.Lenkung_Moment = Data(:,64);

    % displacement / angles:
    outdata.Longitudinal_Weg = Data(:,7); % check: DATA.channels
    outdata.Lateral_Weg = Data(:,18);
    outdata.Vertikal_Weg = Data(:,29);
    outdata.Sturz_Winkel = Data(:,42);
    outdata.Bremse_Winkel = Data(:,53);
    outdata.Lenkung_Winkel = Data(:,63);

    % actuator: KMD, DeltaP, LVDT
    outdata.Longitudinal_DeltaP = Data(:,74); % check: DATA.channels
    outdata.Lateral1_KMD = Data(:,78); 
    outdata.Lateral2_KMD = Data(:,82);
    outdata.Lateral3_KMD = Data(:,87); 
    outdata.Vertikal_DeltaP = Data(:,91);
    outdata.Bremse_DeltaP = Data(:,95);

    outdata.Longitudinal_LVDT = Data(:,75); 
    outdata.Lateral1_LVDT = Data(:,80);
    outdata.Lateral2_LVDT = Data(:,84);
    outdata.Lateral3_LVDT = Data(:,89);
    outdata.Vertikal_LVDT = Data(:,92);
    outdata.Bremse_LVDT = Data(:,96);

    % store actuator signals:
    outdata.FXAK = outdata.Longitudinal_Kraft;
    outdata.FYAK = outdata.Lateral_Kraft;
    outdata.FZAK = outdata.Vertikal_Kraft;
    outdata.MXAK = outdata.Sturz_Moment;
    outdata.MYAK = outdata.Bremse_Moment;
    outdata.MZAK = outdata.Lenkung_Moment;

    outdata.SXAK = outdata.Longitudinal_Weg;
    outdata.SYAK = outdata.Lateral_Weg;
    outdata.SZAK = outdata.Vertikal_Weg;
    outdata.AXAK = outdata.Sturz_Winkel;
    outdata.AYAK = outdata.Bremse_Winkel;
    outdata.AZAK = outdata.Lenkung_Winkel;

    % measuring wheel:
    outdata.FXMR = Data(:,1);
    outdata.FYMR = Data(:,2);
    outdata.FZMR = Data(:,3);
    outdata.MXMR = Data(:,4);
    outdata.MYMR = Data(:,5);
    outdata.MZMR = Data(:,6);

    %% actuator forces:
    outdata.FX_actuator = 0.1 * outdata.Longitudinal_DeltaP * outdata.Area_X * outdata.scale2kilo;
    outdata.FZ_actuator = 0.1 * outdata.Vertikal_DeltaP * outdata.Faktor_Fz * outdata.scale2kilo;
    outdata.MY_actuator = 0.1 * outdata.Bremse_DeltaP * outdata.Area_Brake * outdata.scale2kilo;
    outdata.LAT1_actuator = outdata.Lateral1_KMD;
    outdata.LAT2_actuator = outdata.Lateral2_KMD;
    outdata.LAT3_actuator = outdata.Lateral3_KMD;
end