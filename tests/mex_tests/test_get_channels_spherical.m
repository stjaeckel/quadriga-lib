function test_get_channels_spherical

% Generate test antenna
ant = quadriga_lib.arrayant_generate('omni');
ant.e_theta_re(:,:,2) = 2;
ant.e_theta_im(:,:,2) = 0;
ant.e_phi_re(:,:,2) = 0;
ant.e_phi_im(:,:,2) = 0;
ant.element_pos = [0,0; 1,-1; 0,0];
ant.coupling_re = eye(2);
ant.coupling_im = zeros(2);

fbs_pos = [10,0 ; 0,10 ; 1 11];
path_gain = [1,0.25];
path_length = [0,0];
M = [1,1 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; -1,-1 ; 0,0];

[coeff_re, coeff_im, delay, aod, eod, aoa, eoa] = quadriga_lib.get_channels_spherical( ant, ant, ...
    fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1, 0 );

aod = aod * 180/pi;
eod = eod * 180/pi;
aoa = aoa * 180/pi;
eoa = eoa * 180/pi;

alpha = atan(2.0 / 20.0) * 180.0 / pi;
beta = 90.0;
assertElementsAlmostEqual( aod(:,:,1), [0,alpha;-alpha,0], 'absolute', 1e-14 );
assertElementsAlmostEqual( aod(:,:,2), [beta,beta;beta,beta], 'absolute', 1e-14 );

alpha = 180 - alpha;
beta = 180;
assertElementsAlmostEqual( cosd(aoa(:,:,1)), cosd([beta,-alpha;alpha,beta]), 'absolute', 1e-14 );

alpha = 180.0 - atan(9.0 / 20.0) * 180.0 / pi;
beta = 180.0 - atan(11.0 / 20.0) * 180.0 / pi;
assertElementsAlmostEqual( aoa(:,:,2), [alpha,alpha;beta,beta], 'absolute', 1e-14 );

alpha = atan(10.0 / 11.0) * 180.0 / pi;
beta = atan(10.0 / 9.0) * 180.0 / pi;
assertElementsAlmostEqual( eod(:,:,1), [0,0;0,0], 'absolute', 1e-14 );
assertElementsAlmostEqual( eod(:,:,2), [beta,alpha;beta,alpha], 'absolute', 1e-13 );

alpha = atan(10.0 / sqrt(9.0 * 9.0 + 20.0 * 20.0)) * 180.0 / pi;
beta = atan(10.0 / sqrt(11.0 * 11.0 + 20.0 * 20.0)) * 180.0 / pi;
assertElementsAlmostEqual( eoa(:,:,1), [0,0;0,0], 'absolute', 1e-14 );
assertElementsAlmostEqual( eoa(:,:,2), [alpha,alpha;beta,beta], 'absolute', 1e-13 );

amp = coeff_re.^2 + coeff_im.^2;
assertElementsAlmostEqual( amp(:,:,1), [1,4;4,16], 'absolute', 1e-13 );
assertElementsAlmostEqual( amp(:,:,2), [0.25,1;1,4], 'absolute', 1e-13 );

C = 299792458.0;
d0 = 20.0;
d1 = sqrt(20.0 * 20.0 + 2.0 * 2.0);
e0 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e1 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
e2 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e3 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));

assertElementsAlmostEqual( delay(:,:,1), [d0,d1;d1,d0]/C, 'absolute', 1e-13 );
assertElementsAlmostEqual( delay(:,:,2), [e0,e2;e1,e3]/C, 'absolute', 1e-13 );

% Exception handling
[~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'element_pos');
[~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'coupling_re');
try
    [~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Transmit antenna: Imaginary part of coupling matrix (phase component) defined without real part (absolute component)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant = rmfield(ant,'coupling_im');
[~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

% Mismatching n_path
try
    [~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos(:,[1,2,2]), fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs ''fbs_pos'', ''lbs_pos'', ''path_gain'', ''path_length'', and ''M'' must have the same number of columns (n_paths)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Wrong size of TX position
try
    [~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tx_pos'' has incorrect number of elements.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% B-003: NLOS delay and phase follow path_length when it is longer than the TX-FBS-LBS-RX polyline
% 3-bounce path via (30,40,0), (50,60,0), (70,40,0), TX (0,0,0), RX (100,0,0): 100 + 40*sqrt(2) m, polyline 140 m
ant = quadriga_lib.arrayant_generate('omni');
C = 299792458.0;
fc = 3.0e9;
L_true = 100.0 + 40.0 * sqrt(2.0);
for L = [L_true, 0.0]
    [cr, ci, dl] = quadriga_lib.get_channels_spherical( ant, ant, [30;40;0], [70;40;0], 1, L, [1;0;0;0;0;0;-1;0], ...
        [0;0;0], [0;0;0], [100;0;0], [0;0;0], fc, 1);
    L_exp = max(L, 140.0);
    assertElementsAlmostEqual( dl * C, L_exp, 'absolute', 1e-6 );
    assertElementsAlmostEqual( cr, cos(2*pi*fc/C * L_exp), 'absolute', 1e-6 );
    assertElementsAlmostEqual( ci, -sin(2*pi*fc/C * L_exp), 'absolute', 1e-6 );
end

% 0 paths with add_fake_los_path: one zero-power LOS path with the LOS delay (as in C++ and Python)
ant1 = quadriga_lib.arrayant_generate('omni');
e = zeros(0, 1);
[cr, ci, dl, ad, ed, aa, ea] = quadriga_lib.get_channels_spherical(ant1, ant1, zeros(3,0), zeros(3,0), e, e, zeros(8,0), [0;0;0], [0;0;0], [10;0;0], [0;0;0], 1e9, true, true);
assertEqual( [cr, ci], [0, 0] );
assertElementsAlmostEqual( dl, 10 / C, 'absolute', 1e-18 );
assertElementsAlmostEqual( [ad, ed, abs(aa), ea], [0, 0, pi, 0], 'absolute', 1e-14 );   % aa = +/- pi
[~, ~, dl] = quadriga_lib.get_channels_spherical(ant1, ant1, zeros(3,0), zeros(3,0), e, e, zeros(8,0), [0;0;0], [0;0;0], [10;0;0], [0;0;0], 1e9, false, true);
assertEqual( dl, 0 );
% Per antenna pair: the delays and angles of a real LOS path (moved to slot 0) with the same geometry
ant2 = ant1; ant2.e_theta_re(:,:,2) = 1; ant2.e_theta_im(:,:,2) = 0; ant2.e_phi_re(:,:,2) = 0; ant2.e_phi_im(:,:,2) = 0;
ant2.element_pos = [0.3, -0.2; 0.1, 0.4; 0, 0.2]; ant2.coupling_re = [1, 0.3; 0, 1]; ant2.coupling_im = [0, 0.2; 0, 0];
tx = [1;2;3]; rx = [11;-3;5];
for abs_dl = [true, false]
    o0 = cell(1, 7); oL = cell(1, 7);
    [o0{:}] = quadriga_lib.get_channels_spherical(ant2, ant2, zeros(3,0), zeros(3,0), e, e, zeros(8,0), tx, [0;0;0.3], rx, [0;0;-0.2], 1e9, abs_dl, true);
    [oL{:}] = quadriga_lib.get_channels_spherical(ant2, ant2, (tx+rx)/2, (tx+rx)/2, 1, norm(rx-tx), [1;0;0;0;0;0;-1;0], tx, [0;0;0.3], rx, [0;0;-0.2], 1e9, abs_dl, true);
    assertEqual( size(o0{3}), [2, 2] );
    assertEqual( [o0{1}, o0{2}], zeros(2, 4) );
    for k = 3:7
        assertElementsAlmostEqual( o0{k}, oL{k}(:,:,1), 'absolute', 1e-14 );
    end
end

% 0 paths without add_fake_los_path: empty outputs [n_rx_ports, n_tx_ports, 0] (as in C++ and Python)
o0 = cell(1, 7);
[o0{:}] = quadriga_lib.get_channels_spherical(ant2, ant2, zeros(3,0), zeros(3,0), e, e, zeros(8,0), tx, [0;0;0.3], rx, [0;0;-0.2], 1e9, true, false);
for k = 1:7
    assertEqual( [size(o0{k}, 1), size(o0{k}, 2), size(o0{k}, 3)], [2, 2, 0] );
end

end
