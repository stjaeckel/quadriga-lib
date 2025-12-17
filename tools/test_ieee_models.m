clear all
clc

%% RMS Delay spread

function rds = rms_delay_spread(delay_ns, power_linear)
    % delay_ns, power_linear: vectors (same length)
    P = sum(power_linear);
    w = power_linear ./ P;
    mean_d = dot(w, delay_ns);
    mean_sq = dot(w, delay_ns.^2);
    rds = sqrt(mean_sq - mean_d * mean_d);
end

ChannelType = {'B','C','D','E','F'};

rds = zeros(2,5)
for n = 1:5
    freq = 2.4e9;
    ant = quadriga_lib.arrayant_generate('omni', 30);
    
    % TGn
    ch_n = quadriga_lib.get_channels_ieee_indoor(ant, ant, ChannelType{n}, freq);  % TGn
    power_linear = squeeze( ch_n.coeff_re.^2 + ch_n.coeff_im.^2 );
    delay_ns = squeeze( ch_n.delay * 1e9 );
    rds(1,n) = rms_delay_spread(delay_ns, power_linear);

    % TGac
    ch_ac = quadriga_lib.get_channels_ieee_indoor(ant, ant, ChannelType{n}, freq, 5e-9);  % TGac, 5 ns
    power_linear = squeeze( ch_ac.coeff_re.^2 + ch_ac.coeff_im.^2 );
    delay_ns = squeeze( ch_ac.delay * 1e9 );
    rds(2,n) = rms_delay_spread(delay_ns, power_linear);
end

rds

%% TGn Simulated MIMO Channel Properties

function C = mimo_capacity_logdet(Hf, snr_lin)
    [Nr, Nt, N] = size(Hf);
    scale = sqrt(mean(abs(Hf(:)).^2));   % one scale for all carriers
    if scale > 0, Hf = Hf ./ scale; end
    C = zeros(N,1);
    for k = 1:N
        s = svd(Hf(:,:,k));
        C(k) = sum(log2(1 + (snr_lin/Nt) * (s.^2)));
    end
end

freq = 2.4e9;
spacing = 0.50; 
ant = quadriga_lib.arrayant_generate('ula', 30, freq, [], [], [], 1, 4, [], [], spacing);

ChannelType = {'A', 'B','C','D','E','F','IID'};

pilot_grid = 0;      % Summation of all taps @ fc
bandwidth = 100e6;
n_iterations = 2000;
snr = 10;
dist = 50.0; % ALL NLOS

C = zeros(7,n_iterations);
for n = 1:7
    for m = 1:n_iterations
        if n < 7
            ch = quadriga_lib.get_channels_ieee_indoor(ant, ant, ChannelType{n}, freq, ...
                [], [], [], [], [], [], dist);
            [ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( ch.coeff_re, ch.coeff_im, ...
                ch.delay, pilot_grid, bandwidth );
            H = complex(hmat_re, hmat_im);
        else % IID
            H = (randn(4,4,numel(pilot_grid)) + 1j*randn(4,4,numel(pilot_grid))) / sqrt(2);
        end
        C(n,m) = mean( mimo_capacity_logdet(H,snr) ); % Average over frequency
    end
end

clc
avg_cap = mean(C,2);
percent = round(avg_cap./avg_cap(end) * 100);

[ round(avg_cap,1), percent ]

[ cdf , bins ] = qf.acdf( C , [7:0.01:13] , 2   );

set(0,'defaultTextFontSize', 24)                                        % Default Font Size
set(0,'defaultAxesFontSize', 24)                                        % Default Font Size
set(0,'defaultAxesFontName','Times')                                    % Default Font Type
set(0,'defaultTextFontName','Times')                                    % Default Font Type

figure('Position',[ 100 , 100 , 2*760 , 2*400]);
plot(bins,cdf,'LineWidth',2)
legend(ChannelType,'Location','northwest')
xlabel('Capacity (bps/Hz)')
ylabel('CDF')
grid on


%% Doppler power spectrum
close all
clear all

set(0,'defaultTextFontSize', 24)                                        % Default Font Size
set(0,'defaultAxesFontSize', 24)                                        % Default Font Size
set(0,'defaultAxesFontName','Times')                                    % Default Font Type
set(0,'defaultTextFontName','Times')                                    % Default Font Type

ant = quadriga_lib.arrayant_generate('omni', 30);

n_users = 1;                        % single user channel (TGn)
freq = 2.4e9;                       % Hz
tap_spacing_s = 10e-9;              % seconds
observation_time_s = 2;             % seconds
update_rate_s = 0.02;               % seconds
n_subpath = 100;                    % number of sub-paths
seed = 1234;                        % random seed
bw = 100e6;                         % Bandwidth for Doppler-Delay profile
pilot_grid = -0.5 : 0.01 : 0.5;     % Carriers 0 = freq, 1 = freq + bandwidth
n_iterations = 1000;

n_snap = round( observation_time_s/update_rate_s + 1 );
Doppler_axis_Hz = ((0:n_snap-1)/n_snap - 0.5) / update_rate_s;

% TGn environment motion only, model C
v_sta_kmh = 0;       % station speed
v_env_kmh = 1.2;     % TGn enviroment speed

Doppler_profile = zeros(n_snap, n_iterations);
for n = 1 : n_iterations
    ch = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'C', freq, tap_spacing_s, n_users, ...
        observation_time_s, update_rate_s, v_sta_kmh, v_env_kmh, [], [], [], [], n_subpath, [], seed+n );
    [ Hre, Him ] = quadriga_lib.baseband_freq_response( ch.coeff_re, ch.coeff_im, ch.delay, pilot_grid, bw );
    H = permute( complex(Hre,Him),[3,4,1,2] );         % Reorder dimensions
    G = fftshift(ifft2(H),2);                          % 2D IFFT
    Doppler_profile(:,n) = sum( abs(G).^2 , 1 )'; 
end
avg_Doppler_dB = 10*log10(mean(Doppler_profile,2));
all_Doppler_dB = 10*log10(Doppler_profile);

figure('Position',[ 100 , 100 , 2*760 , 2*400]);
plot(Doppler_axis_Hz, all_Doppler_dB, '.g')
hold on
plot(Doppler_axis_Hz, avg_Doppler_dB, '-k', 'LineWidth',2)
hold off
grid on
xlabel("Doppler shift (Hz)")
ylabel("Power (dB)")
title("Doppler spectrum, Model C, v_{env} = 1.2 km/h")
saveas(gcf,'Doppler_C1.png')

% TGn station motion only, model C
v_sta_kmh = 5;       % station speed
v_env_kmh = 0;     % TGn enviroment speed

Doppler_profile = zeros(n_snap, n_iterations);
for n = 1 : n_iterations
    ch = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'C', freq, tap_spacing_s, n_users, ...
        observation_time_s, update_rate_s, v_sta_kmh, v_env_kmh, [], [], [], [], n_subpath, [], seed+n );
    [ Hre, Him ] = quadriga_lib.baseband_freq_response( ch.coeff_re, ch.coeff_im, ch.delay, pilot_grid, bw );
    H = permute( complex(Hre,Him),[3,4,1,2] );         % Reorder dimensions
    G = fftshift(ifft2(H),2);                          % 2D IFFT
    Doppler_profile(:,n) = sum( abs(G).^2 , 1 )'; 
end
avg_Doppler_dB = 10*log10(mean(Doppler_profile,2));
all_Doppler_dB = 10*log10(Doppler_profile);

figure('Position',[ 100 , 100 , 2*760 , 2*400]);
plot(Doppler_axis_Hz, all_Doppler_dB, '.g')
hold on
plot(Doppler_axis_Hz, avg_Doppler_dB, '-k', 'LineWidth',2)
hold off
grid on
xlabel("Doppler shift (Hz)")
ylabel("Power (dB)")
title("Doppler spectrum, Model C, v_{sta} = 5.0 km/h")
saveas(gcf,'Doppler_C2.png')

% Vehicle motion model F
v_sta_kmh = 0;       % station speed
v_env_kmh = 0.089;   % TGac enviroment speed
v_veh_kmh = 40.0;    % Vehicle speed

observation_time_s = 1;             % seconds
update_rate_s = 0.004;               % seconds
n_snap = round( observation_time_s/update_rate_s + 1 );
Doppler_axis_Hz = ((0:n_snap-1)/n_snap - 0.5) / update_rate_s;

Doppler_profile = zeros(n_snap, n_iterations);
Doppler_profile_tap3 = zeros(n_snap, n_iterations);
for n = 1 : n_iterations
    ch = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'F', freq, tap_spacing_s, n_users, ...
        observation_time_s, update_rate_s, v_sta_kmh, v_env_kmh, [], [], [], [], n_subpath, v_veh_kmh, seed+n );
    [ Hre, Him ] = quadriga_lib.baseband_freq_response( ch.coeff_re, ch.coeff_im, ch.delay, pilot_grid, bw );
    H = permute( complex(Hre,Him),[3,4,1,2] );         % Reorder dimensions
    G = fftshift(ifft2(H),2);                          % 2D IFFT
    Doppler_profile(:,n) = sum( abs(G).^2 , 1 )'; 

    [ Hre, Him ] = quadriga_lib.baseband_freq_response( ch.coeff_re(:,:,4,:), ch.coeff_im(:,:,4,:), ch.delay(:,:,4,:), pilot_grid, bw );
    H = permute( complex(Hre,Him),[3,4,1,2] );         % Reorder dimensions
    G = fftshift(ifft2(H),2);                          % 2D IFFT
    Doppler_profile_tap3(:,n) = sum( abs(G).^2 , 1 )'; 
end
avg_Doppler_dB = 10*log10(mean(Doppler_profile,2));
all_Doppler_dB = 10*log10(Doppler_profile);
avg_Doppler_dB_tap3 = 10*log10(mean(Doppler_profile_tap3,2));

figure('Position',[ 100 , 100 , 2*760 , 2*400]);
plot(Doppler_axis_Hz, all_Doppler_dB, '.g')
hold on
plot(Doppler_axis_Hz, avg_Doppler_dB, '-k', 'LineWidth',2)
plot(Doppler_axis_Hz, avg_Doppler_dB_tap3, '--r', 'LineWidth',2)
hold off
grid on
xlabel("Doppler shift (Hz)")
ylabel("Power (dB)")
title("Doppler spectrum, Model F, v_{env} = 0.089 km/h, v_{veh} = 40.0 km/h")
saveas(gcf,'Doppler_F.png')

% Fluorescent modulation model E
v_sta_kmh = 0;       % station speed
v_env_kmh = 1.2;     % TGn enviroment speed
mains_freq = 50.0;   % Hz

observation_time_s = 0.4;            % seconds
update_rate_s = 0.0008;              % seconds
n_snap = round( observation_time_s/update_rate_s + 1 );
Doppler_axis_Hz = ((0:n_snap-1)/n_snap - 0.5) / update_rate_s;

Doppler_profile = zeros(n_snap, n_iterations);
for n = 1 : n_iterations
    ch = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'E', freq, tap_spacing_s, n_users, ...
        observation_time_s, update_rate_s, v_sta_kmh, v_env_kmh, [], [], [], [], n_subpath, mains_freq, seed+n );
    [ Hre, Him ] = quadriga_lib.baseband_freq_response( ch.coeff_re, ch.coeff_im, ch.delay, pilot_grid, bw );
    H = permute( complex(Hre,Him),[3,4,1,2] );         % Reorder dimensions
    G = fftshift(ifft2(H),2);                          % 2D IFFT
    Doppler_profile(:,n) = sum( abs(G).^2 , 1 )'; 
end
avg_Doppler_dB = 10*log10(mean(Doppler_profile,2));
all_Doppler_dB = 10*log10(Doppler_profile);

figure('Position',[ 100 , 100 , 2*760 , 2*400]);
plot(Doppler_axis_Hz, all_Doppler_dB, '.g')
hold on
plot(Doppler_axis_Hz, avg_Doppler_dB, '-k', 'LineWidth',2)
hold off
grid on
xlabel("Doppler shift (Hz)")
ylabel("Power (dB)")
title("Doppler spectrum, Model E, v_{env} = 1.2 km/h, f_{m} = 50.0 Hz")
saveas(gcf,'Doppler_E.png')


%% Angles
ant = quadriga_lib.arrayant_generate('custom', 1, 2.4e9, 6, 10);
ant = quadriga_lib.arrayant_copy_element(ant, 1, 2:36);
for n = 2 : 36
    ant = quadriga_lib.arrayant_rotate_pattern(ant,0,0,(n-1)*10, 0,n);
end

of = ones(4,2) * 100;
c = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'A', 2.4e9, 10e-9, 2, [], [], [], [], 1.99, [], [], of, 100, 11 );
P1 = c(1).coeff_re(:,:,2).^2 + c(1).coeff_im(:,:,2).^2; 
P2 = c(2).coeff_re(:,:,2).^2 + c(2).coeff_im(:,:,2).^2; 

figure(1)
bar(sum(P1,1));

figure(2)
bar(sum(P1,2));

figure(3)
bar(sum(P2,1));

figure(4)
bar(sum(P2,2));


 
% W = pow(:,:);
% A = aod(:,:);
% 
% AS_rms = sqrt( sum(W.*(A.^2),1) ./ sum(W,1) ); % 1×M
% 
% stem(aod(:,1), pow(:,1))
% 
% P = 10*log10(squeeze(sum(pow,1)));
% imagesc(P)


n_users = 6;
seed_AoD_LOS = 2803;
range_AoD_LOS = 360;
rand('seed',seed_AoD_LOS);
offsets_AoD_LOS = (rand(n_users,1)-0.5)*range_AoD_LOS

%% Jakes Doppler
close all
clear all
clc

fGHz = 2.4;
tap_spacing_ns = 10;
onservation_time_s = 2;
update_rate_s = 0.0005;
speed_kmh = 50;

ant = quadriga_lib.arrayant_generate('omni');
c = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'F', fGHz, tap_spacing_ns, 1, onservation_time_s, update_rate_s, 2, 0.089, [], [], [], [], 100, speed_kmh, 1234 );

w = size(c.coeff_re,4);
Delta_t = update_rate_s;                 % snapshot spacing
Doppler_axis_Hz = ((0:w-1)/w - 0.5) / Delta_t;

pilot_grid = 0 : 0.01 : 1;
BW = 100e6;    

[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( c.coeff_re, c.coeff_im, c.delay, pilot_grid, BW );
H = complex(hmat_re, hmat_im);
H = permute( H,[3,4,1,2] );                     % Reorder dimensions
G = ifft2(H);                                   % 2D IFFT
G = fftshift( G,2);                             % Center Doppler spectrum

DS = 10*log10( sum( abs(G).^2 , 1 )' );

figure
plot(Doppler_axis_Hz, DS)
xlabel("Doppler shift (Hz)")
ylabel("Power (dB)")


%%
pilot_grid = 0 : 0.01 : 1;

no_snap = size(c.coeff_re,4);

w  = 1000;                                               % Doppler analysis windows size (100 ms)
BW = 100e6;                                             % Channel bandwidth (100 MHz)
N  = 256;                                               % Number of carriers

Doppler_axis = -( (0:w-1)/(w-1)-0.5)/update_rate;     % The Doppler axis in Hz
time = ( 0 : no_snap-1 ) * update_rate; 
Time_axis = time( 1:w:end );                            % Time axis in seconds

no_Doppler = floor( numel(time) ./ w );                 % Number of Doppler samples


    Doppler_spectrum = zeros( w, no_Doppler );          % Preallocate Memory
    for n = 1 : floor( numel(time) ./ w )
        ind = (n-1)*w + 1 : n*w;                        % Snapshot indices

        [ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( c.coeff_re, c.coeff_im, c.delay, pilot_grid, BW, ind );
        H = complex(hmat_re, hmat_im);

        H = permute( H,[3,4,1,2] );                     % Reorder dimensions
        G = ifft2(H);                                   % 2D IFFT
        G = fftshift( G,2);                             % Center Doppler spectrum
        Doppler_spectrum( :,n ) = 10*log10( sum( abs(G).^2 , 1 )' );    % Logrithmic power
    end
    
    figure('Position',[ 100 , 100 , 2*760 , 2*400]);        % New figure
    imagesc(Time_axis,Doppler_axis,Doppler_spectrum);   % Create images    
    colorbar
    title(['Doppler Spectrum',]);
    xlabel('Time [s]'); ylabel('Doppler shift [Hz]');
    set(gca,'Ydir','Normal')                            % Invert y axis
    colormap jet

