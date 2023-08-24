
clear all

%RandStream.setGlobalStream(RandStream('mt19937ar','seed',1));

a = qd_arrayant('omni');
a.Fa(:) = 1;
a.Fb(:) = 1;

% a.copy_element(1,3);
% a.Fa(:,:,2) = 0;
% a.coupling = ones(3,2); %[1;1;1];% [1,0,0 ; 1,0,0 ; 1,0,0 ];;

if 1
    b = qd_builder('LOSonly');
else
    b = qd_builder('3GPP_38.901_UMi_LOS_GR');
end


b.simpar.use_3GPP_baseline = 0;
b.simpar.center_frequency = 299792458.0;
% b.scenpar.NumSubPaths = 1;
b.simpar.use_absolute_delays = 1;
b.simpar.use_random_initial_phase = 0;
b.simpar.show_progress_bars = 0;


b.tx_position = [0;0;1];
b.tx_array = a;
b.rx_array = a;

if 1
    b.rx_positions = [20;0;1];
else
    b.rx_track = qd_track('linear',1,0);
    b.rx_track.interpolate_positions(10);
    b.rx_track.initial_position = [20;0;1];
end

b.check_dual_mobility;
% b.tx_track.orientation(1) = -pi/2;
% b.tx_track.orientation(2) = pi/2;
b.tx_track.orientation(3) = 0; %pi/2;

%b.rx_track.orientation(1) = pi/2;


b.gen_parameters;

b.add_sdc([0;10;1],-6,'absolute','absolute',1000,0,1);

b.xprmat(:,2) = rand(4,1)*1j+rand(4,1)

if b.NumClusters > 1
    %b.pow(2) = 0;
end

c = b.get_channels_old;

d = b.get_channels;

%squeeze( c.delay - d.delay )

%squeeze( c.coeff - d.coeff )

squeeze(c.coeff)

squeeze(d.coeff)

