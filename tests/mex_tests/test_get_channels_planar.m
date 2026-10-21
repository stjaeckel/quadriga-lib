function test_get_channels_planar

% Generate test antenna
ant = quadriga_lib.arrayant_generate('omni');
ant.e_theta_re(:,:,2) = 2;
ant.e_theta_im(:,:,2) = 0;
ant.e_phi_re(:,:,2) = 0;
ant.e_phi_im(:,:,2) = 0;
ant.element_pos = [0,0; 1,-1; 0,0];
ant.coupling_re = eye(2);
ant.coupling_im = zeros(2);

aod = [0, 90] * pi/180;
eod = [0, 45] * pi/180;
aoa = [180, 180 - atand(10/20)] * pi/180;
eoa = [0, atand(10 / sqrt(10^2 + 20^2) )] * pi/180;

path_gain = [1,0.25];
path_length = [20,  sqrt(10^2 + 10^2) + sqrt(10^2 + 20^2 + 10^2)  ];
M = [1,1 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; -1,-1 ; 0,0];

[coeff_re, coeff_im, delay, rx_Doppler] = quadriga_lib.get_channels_planar( ant, ant, ...
    aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

amp = coeff_re.^2 + coeff_im.^2;
assertElementsAlmostEqual( amp(:,:,1), [1,4;4,16], 'absolute', 1e-13 );
assertElementsAlmostEqual( amp(:,:,2), [0.25,1;1,4], 'absolute', 1e-13 );

C = 299792458.0;
d0 = 20.0;
d1 = 20.0;
e0 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e1 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
e2 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e3 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));

assertElementsAlmostEqual( delay(:,:,1), [d0,d1;d1,d0]/C, 'absolute', 1e-13 );
assertElementsAlmostEqual( delay(:,:,2), [e0,e2;e1,e3]/C, 'absolute', 1.2e-10 );

Doppler = cos(aoa(2)) * cos(eoa(2));
assertElementsAlmostEqual( rx_Doppler, [-1, Doppler], 'absolute', 1e-13 );

% Rectangular coupling with omitted imaginary part (regression: the default
% coupling_im must match coupling_re's shape, not n_elements x n_elements).
% Omitting coupling_im must behave identically to a matching-shape zero block.
ant_a = ant;
ant_a.coupling_re = [1; 1];        % 2 elements -> 1 port
ant_a = rmfield(ant_a, 'coupling_im');

ant_b = ant_a;
ant_b.coupling_im = [0; 0];        % explicit matching-shape zero imaginary part

[cra, cia, dla] = quadriga_lib.get_channels_planar( ant_a, ant_a, ...
    aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 0.0, 1);
[crb, cib, dlb] = quadriga_lib.get_channels_planar( ant_b, ant_b, ...
    aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 0.0, 1);

assertEqual( size(cra), [1, 1, 2] );
assertElementsAlmostEqual( cra, crb, 'absolute', 1e-14 );
assertElementsAlmostEqual( cia, cib, 'absolute', 1e-14 );
assertElementsAlmostEqual( dla, dlb, 'absolute', 1e-14 );

% Exception handling
[~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'element_pos');
[~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'coupling_re');
try
    [~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Transmit antenna: Imaginary part of coupling matrix (phase component) defined without real part (absolute component)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant = rmfield(ant,'coupling_im');
[~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

% Mismatching n_path
try
    [~] = quadriga_lib.get_channels_planar( ant, ant, aod([1,2,2]), eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs ''aod'', ''eod'', ''aoa'', ''eoa'', ''path_gain'', ''path_length'', and ''M'' must have the same number of columns (n_paths)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Wrong size of TX position
try
    [~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1;2], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tx_pos'' has incorrect number of elements.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% 0 paths with add_fake_los_path: one zero-power LOS path with the LOS delay (as in C++ and Python)
C = 299792458.0;
ant1 = quadriga_lib.arrayant_generate('omni');
e = zeros(0, 1);
[cr, ci, dl, dop] = quadriga_lib.get_channels_planar(ant1, ant1, e, e, e, e, e, e, zeros(8,0), [0;0;0], [0;0;0], [10;0;0], [0;0;0], 1e9, true, true);
assertEqual( [cr, ci, dop], [0, 0, 0] );
assertElementsAlmostEqual( dl, 10 / C, 'absolute', 1e-18 );
[~, ~, dl] = quadriga_lib.get_channels_planar(ant1, ant1, e, e, e, e, e, e, zeros(8,0), [0;0;0], [0;0;0], [10;0;0], [0;0;0], 1e9, false, true);
assertEqual( dl, 0 );
% Per antenna pair: the delays of a real LOS path (moved to slot 0) with the same geometry
ant2 = ant1; ant2.e_theta_re(:,:,2) = 1; ant2.e_theta_im(:,:,2) = 0; ant2.e_phi_re(:,:,2) = 0; ant2.e_phi_im(:,:,2) = 0;
ant2.element_pos = [0.3, -0.2; 0.1, 0.4; 0, 0.2]; ant2.coupling_re = [1, 0.3; 0, 1]; ant2.coupling_im = [0, 0.2; 0, 0];
tx = [1;2;3]; rx = [11;-3;5]; d = rx - tx;
az = atan2(d(2), d(1)); el = atan2(d(3), hypot(d(1), d(2)));
for abs_dl = [true, false]
    [cr0, ci0, dl0] = quadriga_lib.get_channels_planar(ant2, ant2, e, e, e, e, e, e, zeros(8,0), tx, [0;0;0.3], rx, [0;0;-0.2], 1e9, abs_dl, true);
    [~, ~, dlL] = quadriga_lib.get_channels_planar(ant2, ant2, az, el, atan2(-d(2), -d(1)), -el, 1, norm(d), [1;0;0;0;0;0;-1;0], tx, [0;0;0.3], rx, [0;0;-0.2], 1e9, abs_dl, true);
    assertEqual( size(dl0), [2, 2] );
    assertEqual( [cr0, ci0], zeros(2, 4) );
    assertElementsAlmostEqual( dl0, dlL(:,:,1), 'absolute', 1e-17 );
end

% 0 paths without add_fake_los_path: empty outputs [n_rx_ports, n_tx_ports, 0] (as in C++ and Python)
[cr, ci, dl, dop] = quadriga_lib.get_channels_planar(ant2, ant2, e, e, e, e, e, e, zeros(8,0), tx, [0;0;0.3], rx, [0;0;-0.2], 1e9, true, false);
assertEqual( [size(cr, 1), size(cr, 2), size(cr, 3)], [2, 2, 0] );
assertEqual( size(ci), size(cr) );
assertEqual( size(dl), size(cr) );
assertTrue( isempty(dop) );

end
