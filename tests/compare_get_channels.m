
clear all

a = qd_arrayant('omni');
a.copy_element(1,3);
a.Fa(:,:,2) = 0;
a.coupling = [1,0,0 ; 1,0,0 ; 1,0,0 ];

b = qd_builder('LOSonly');
b.simpar.center_frequency = 2997924580.0;
b.scenpar.NumSubPaths = 1;
b.simpar.use_absolute_delays = 0;
b.simpar.use_random_initial_phase = 0;


b.tx_position = [0;0;1];
b.rx_positions = [20;0;1];
b.tx_array = a;
b.rx_array = a;

b.check_dual_mobility;
% b.tx_track.orientation(1) = -pi/2;
% b.tx_track.orientation(2) = pi/2;
b.tx_track.orientation(3) = pi/2;

%b.rx_track.orientation(1) = pi/2;


b.gen_parameters;
%b.add_sdc([0;10;11],-6,'absolute','absolute',1000,0,1);



c = b.get_channels;

c.coeff
c.delay




